# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import os
import json
import time
import datetime
from pathlib import Path
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from dist_utils import main_process, is_dist_avail_and_initialized, is_main_process, get_rank, get_world_size
from logger import MetricLogger, SmoothedValue
from utils import prepare_sample
from optims import get_optimizer, LinearWarmupCosineLRScheduler


class Runner:
    def __init__(self, cfg, model, train_loader, valid_loader, job_id):
        self.config = cfg
        self.run_cfg = cfg.get("run", {})  # 분산/학습 관련 설정 블록

        # log dir
        out_root = self.run_cfg.get("output_dir", "./outputs")
        self.output_dir = Path(out_root) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # model storage path
        self.model_storage_path = Path(self.run_cfg.get("model_storage_path", "./model_storage"))
        self.model_storage_path.mkdir(parents=True, exist_ok=True)

        # settings
        self.device = torch.device(self.run_cfg.get("device", cfg.get("device", "cuda")))
        self.use_distributed = bool(self.run_cfg.get("use_distributed", False))
        self.start_epoch = int(self.run_cfg.get("start_epoch", 0))
        opt_cfg = self.run_cfg.get("optims", {})
        self.max_epoch = int(opt_cfg.get("max_epoch", 10))
        self.evaluate_only = bool(self.run_cfg.get("evaluate", False))
        self.cuda_enabled = (self.device.type == "cuda")

        # model / DDP
        self._model = model.to(self.device)
        if self.use_distributed:
            # run_cfg["gpu"] 에 rank별 local gpu id가 들어있다고 가정
            gpu_id = self.run_cfg.get("gpu", 0)
            self.model = DDP(self._model, device_ids=[gpu_id])
        else:
            self.model = self._model

        # dataloaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader    

        # AMP
        self.use_amp = bool(self.run_cfg.get("amp", False))
        self.scaler = GradScaler(enabled=self.use_amp)

        # optimizer & scheduler
        self.iters_per_epoch = (
            len(self.train_loader)
            if self.run_cfg.get("epoch_based", True)
            else int(self.run_cfg.get("iters_per_epoch", len(self.train_loader)))
        )

        self.optimizer = get_optimizer(self.model, opt_cfg)

        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=opt_cfg.get("min_lr", 0.0),
            init_lr=opt_cfg.get("init_lr", opt_cfg.get("peak_lr", 1e-3)),
            warmup_steps=opt_cfg.get("warmup_steps", 0),
            warmup_start_lr=opt_cfg.get("warmup_start_lr", -1),
        )

        self.log_config()

    def unwrap_dist_model(self, model):
        return model.module if self.use_distributed else model

    def train_epoch(self, epoch: int):
        """
        한 epoch 동안 모델을 학습시키는 함수.
        - epoch: 현재 epoch 번호 (0-index 또는 1-index는 호출하는 쪽에서 관리)
        """

        # 1. 모델을 학습 모드로 전환 (Dropout, BatchNorm 등이 train 모드로 동작)
        self.model.train()

        # 2. 학습 과정에서 지표를 기록할 MetricLogger 생성
        metric_logger = MetricLogger(delimiter="  ")
        # 학습률(lr)을 기록할 meter 등록 (소수점 6자리까지 출력)
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        # loss를 기록할 meter 등록 (소수점 4자리까지 출력)
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, self.iters_per_epoch
            )
        )
        header = f"Train: data epoch: [{epoch}]"

        accum_grad_iters = int(self.run_cfg.get("accum_grad_iters", 1))
        log_freq = int(self.run_cfg.get("log_freq", 50))

        iter_loader = iter(self.train_loader)

        # 3. 미니배치 단위 학습 루프
        # metric_logger.log_every:
        #   - range(self.iters_per_epoch)를 순회하면서
        #   - log_freq마다 로그를 찍어주는 헬퍼
        for i in metric_logger.log_every(
            range(self.iters_per_epoch),
            log_freq,
            header=header,
            logger=self.log_writter,
            start_step=epoch * self.iters_per_epoch,
        ):
            try:
                samples = next(iter_loader)
            except StopIteration:
                iter_loader = iter(self.train_loader)
                samples = next(iter_loader)

            # 3-2. 샘플을 GPU로 옮기거나 필요한 전처리 수행
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            # 3-3. 스케줄러로 현재 step의 learning rate 업데이트
            #      (epoch, step 기반으로 lr를 조정하는 스케줄러라고 가정)
            self.scheduler.step(cur_epoch=epoch, cur_step=i)

            # 3-4. 자동 혼합 정밀도(AMP) 컨텍스트에서 forward
            #      - self.use_amp == True면 float16/float32 혼합으로 연산 → 속도/메모리 이점
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 모델은 samples를 입력으로 받아, "loss" 키를 포함한 dict를 반환한다고 가정
                outputs = self.model(samples)
                loss = outputs["loss"]

            # 3-5. backward (기울기 계산)
            if self.use_amp:
                # AMP 사용하는 경우: 스케일러로 loss를 스케일링 후 backward
                self.scaler.scale(loss).backward()
            else:
                # AMP 미사용: 일반적인 backward
                loss.backward()

            # 3-6. Gradient Accumulation
            #     - 여러 step 동안 기울기를 누적했다가, 일정 step마다 optimizer.step() 수행
            if (i + 1) % accum_grad_iters == 0:
                if self.use_amp:
                    # 스케일된 grad를 사용하여 optimizer step 수행
                    self.scaler.step(self.optimizer)
                    # 다음 step을 위해 스케일러 상태 업데이트
                    self.scaler.update()
                else:
                    # 일반적인 optimizer step
                    self.optimizer.step()

                # step 후에는 항상 grad를 초기화해 줘야 함
                self.optimizer.zero_grad()

            # 3-7. 로깅을 위한 지표 업데이트
            metric_logger.update(loss=loss.item())
            # optimizer의 첫 번째 param group에서 현재 lr 추출
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        # 4. (분산 학습 시) 모든 프로세스 사이의 metric을 동기화
        metric_logger.synchronize_between_processes()

        logging.info("Averaged stats: " + str(metric_logger.global_avg()))

        # 5. 각 meter의 global_avg를 깔끔한 문자열 형태로 반환
        #    예: {"loss": "0.123", "lr": "0.000100"}
        return {
            name: "{:.3f}".format(meter.global_avg)
            for name, meter in metric_logger.meters.items()
        }



    @torch.no_grad()
    def valid_epoch(self, epoch, split, decode=False, save_json=False):
        model = self.unwrap_dist_model(self.model)
        model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, "{}_loader does not exist.".format(split)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Eval: data epoch: [{}]".format(epoch)
        log_freq = int(self.run_cfg.get("log_freq", 50))

        results = []
        for samples in metric_logger.log_every(dataloader, log_freq, header=header):
            samples = prepare_sample(samples, cuda_enabled=self.cuda_enabled)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                forward_result = model(samples, verbose=True)

            loss = forward_result.get("loss", torch.tensor(0.0, device=self.device))
            correct = forward_result.get("correct", torch.tensor(0.0, device=self.device))
            total = forward_result.get("total", torch.tensor(1.0, device=self.device))

            # ground truth text는 List[str]일 가능성이 높으므로 그대로 저장
            res = {
                "ground_truth": samples.get("text", []),
                "loss": float(loss.item()),
                "acc": float((correct / total).item()),
                "total": int(total.item()),
            }

            if decode:
                text = model.generate(samples, self.run_cfg)
                res["text"] = text

            results.append(res)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if save_json:
            self.save_result(results, self.output_dir, f"eval_{split}_epoch_{epoch}")

        # 집계
        dev = self.device
        res_tensors = {
            "loss": torch.tensor(0.0, device=dev),
            "n_sample": torch.tensor(0.0, device=dev),
            "correct": torch.tensor(0.0, device=dev),
            "n_token": torch.tensor(0.0, device=dev),
        }

        for item in results:
            item_loss = float(item["loss"])
            item_total = float(item["total"])
            item_acc = float(item["acc"])

            # 여기서는 loss를 "토큰 수 기준 가중 평균"으로 합산한다고 가정
            item_n_sample = item_total  # 또는 1.0 으로 두고 샘플 평균으로 볼 수도 있음
            item_correct = item_acc * item_total
            item_n_token = item_total

            res_tensors["loss"] += item_loss * item_n_sample
            res_tensors["n_sample"] += item_n_sample
            res_tensors["correct"] += item_correct
            res_tensors["n_token"] += item_n_token

        if is_dist_avail_and_initialized():
            for k in res_tensors:
                dist.all_reduce(res_tensors[k])

        ret = {"loss": 0.0, "agg_metrics": 0.0}
        if res_tensors["n_sample"] > 0:
            ret["loss"] = (res_tensors["loss"] / res_tensors["n_sample"]).item()
        if res_tensors["n_token"] > 0:
            ret["agg_metrics"] = (res_tensors["correct"] / res_tensors["n_token"]).item()

        return ret


    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(result, open(final_result_file, "w", encoding="utf-8"), ensure_ascii=False)

            print("result file saved to %s" % final_result_file)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0.0
        best_val_loss = float('inf')
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(cur_epoch, "valid", decode=False, save_json=False)
            
            val_loss = None
            if valid_log is not None and is_main_process():
                agg_metrics = valid_log["agg_metrics"]
                val_loss = valid_log["loss"]
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = cur_epoch
                    self.save_checkpoint(cur_epoch, is_best=True, val_loss=val_loss)
                
                # Also save if it's the best metric (optional, keeping existing logic)
                if agg_metrics > best_agg_metric:
                    best_agg_metric = agg_metrics
                    # self.save_checkpoint(cur_epoch, is_best=True) # Already saved if loss is best, or we can save separately

                valid_log.update({"best_epoch": best_epoch, "best_val_loss": best_val_loss})
                self.log_stats(valid_log, split_name="valid")

            # Save periodic checkpoint
            self.save_checkpoint(cur_epoch, is_best=False, val_loss=val_loss)

            if self.use_distributed:
                dist.barrier()


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config, indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {f"{split_name}_{k}": v for k, v in stats.items()}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best: bool = False, val_loss: float = None):
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic and not param_grad_dic[k]:
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": cur_epoch,
            "val_loss": val_loss
        }
        
        # 1. Save to original output_dir (keep existing behavior)
        fname = "checkpoint_{}.pth".format("best" if is_best else cur_epoch)
        save_to = os.path.join(self.output_dir, fname)
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

        # 2. Save to model_storage_path with custom name
        # Format: {experiment_name}_epoch{epoch}_loss{val_loss}.pth
        exp_name = self.config.get("experiment_name", "model")
        loss_str = f"{val_loss:.4f}" if val_loss is not None else "nan"
        
        if is_best:
            storage_fname = f"{exp_name}_best_loss{loss_str}.pth"
        else:
            storage_fname = f"{exp_name}_epoch{cur_epoch:02d}_loss{loss_str}.pth"
            
        storage_save_to = self.model_storage_path / storage_fname
        logging.info("Saving storage checkpoint to {}.".format(storage_save_to))
        torch.save(save_obj, storage_save_to)
