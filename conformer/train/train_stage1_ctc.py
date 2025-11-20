from __future__ import annotations

import math
import time
import random
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

# ★★ 중요: 프로젝트 루트(/data_x/.../conformer)를 sys.path에 추가 ★★
ROOT_DIR = Path(__file__).resolve().parents[1]  # .../conformer
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from tokenizer import TextTokenizer
from data.features import LogMelFeatureExtractor
from data.collate import SpeechDataCollator
from data.dataloader import build_dataset, build_dataloader
from conformer import Conformer  # conformer 패키지 구조에 맞게 필요시 수정
import os
os.environ["HF_DATASETS_AUDIO_TORCHCODEC_DISABLED"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------

torch.backends.cudnn.benchmark = False
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


config: Dict = {
    "experiment_name": "stage1_ctc_conformer",
    "seed": 1337,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data": {
        "sample_rate": 16_000,
        "text_lowercase": True,
        "max_audio_seconds": 20.0,
        "pad_to_max_seconds": True,
        "hf_dataset": {
            "name": "parler-tts/mls_eng",
            "config": None,
            "train_split": "train",
            "valid_split": "dev",
            "text_column": "transcript",
            "audio_column": "audio",
            "cache_dir": "data/hf-cache",
            "streaming": False,
        },
    },
    "feature_extractor": {
        "n_mels": 80,
        "n_fft": 1024,
        "win_length": 400,
        "hop_length": 160,
        "f_min": 0.0,
        "f_max": None,
        "mel_power": 2.0,
        "log_offset": 1e-6,
    },
    "tokenizer": {
        "use_sentencepiece": False,
        "sentencepiece_model": "tokenizer/stage1_sp.model",
        "blank_id": 0,
        "chars": list("abcdefghijklmnopqrstuvwxyz'1234567890-,;.?! "),
    },
    "model": {
        "input_dim": 80,
        "encoder_dim": 512,
        "num_layers": 3,
        "num_attention_heads": 8,
        "feed_forward_expansion_factor": 4,
        "conv_expansion_factor": 2,
        "conv_kernel_size": 31,
        "dropout": 0.1,
        "subsampling_factor": 8,
        "min_subsample_len_multiplier": 2,
    },
    "dataloader": {
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "shuffle": True,
    },
    "optim": {
        "peak_lr": 1e-3,
        "weight_decay": 1e-4,
        "eps": 1e-9,
        "betas": (0.9, 0.98),
        "grad_accum_steps": 4,
    },
    "scheduler": {
        "warmup_steps": 20_000,
        "total_steps": 250_000,
        "final_lr_scale": 0.01,
    },
    "trainer": {
        "num_epochs": 30,
        "log_interval": 50,
        "val_interval": 1,
        "grad_clip": 5.0,
        "use_amp": True,
        "checkpoint_dir": "checkpoints/stage1",
        "max_to_keep": 5,
        "resume_from": None,
    },
}


# ------------------------------------------------------------
# 스케줄러 / 유틸
# ------------------------------------------------------------

class WarmupExponentialDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    warmup 이후 exponential decay로 lr를 줄이는 스케줄러.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        final_lr_scale: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        self.final_lr_scale = final_lr_scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch + 1)

        if step <= self.warmup_steps:
            # linear warmup
            scale = step / self.warmup_steps
        else:
            # exponential decay from 1.0 → final_lr_scale
            progress = min(
                1.0,
                (step - self.warmup_steps)
                / (self.total_steps - self.warmup_steps),
            )
            scale = math.exp(math.log(self.final_lr_scale) * progress)

        return [base_lr * scale for base_lr in self.base_lrs]


class AverageMeter:
    """
    running average 계산용 유틸.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def save_checkpoint(state: Dict, checkpoint_dir: str, max_to_keep: int) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"epoch{state['epoch']:02d}_val{state['val_loss']:.4f}.pt"
    torch.save(state, ckpt_path)

    checkpoints = sorted(checkpoint_dir.glob("epoch*.pt"))
    if len(checkpoints) > max_to_keep:
        for stale in checkpoints[:-max_to_keep]:
            stale.unlink(missing_ok=True)

    return ckpt_path


# ------------------------------------------------------------
# train / eval loop
# ------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupExponentialDecayScheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    grad_accum_steps: int,
    grad_clip: float,
    log_interval: int,
    amp_enabled: bool,
) -> Tuple[int, float]:
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    steps_in_accum = 0
    start_time = time.time()
    skipped_batches = 0

    def _optimizer_step():
        nonlocal global_step, steps_in_accum
        if steps_in_accum == 0:
            return

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1
        steps_in_accum = 0

    for batch_idx, batch in enumerate(dataloader, start=1):
        if batch is None:
            continue

        features, input_lengths, targets, target_lengths, _ = batch
        features = features.to(device)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with autocast(enabled=amp_enabled):
            logits, logit_lengths = model(features, input_lengths)

        if torch.any(target_lengths > logit_lengths):
            skipped_batches += 1
            continue

        with autocast(enabled=amp_enabled):
            loss = criterion(
                logits.transpose(0, 1),  # [T, B, C]
                targets,
                logit_lengths,
                target_lengths,
            )
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        steps_in_accum += 1
        loss_meter.update(loss.item() * grad_accum_steps, n=features.size(0))

        if steps_in_accum == grad_accum_steps:
            _optimizer_step()
            if global_step > 0 and global_step % log_interval == 0:
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Epoch {epoch:02d} | step {global_step} | "
                    f"loss {loss_meter.avg:.4f} | lr {current_lr:.2e} | "
                    f"{elapsed:.1f}s"
                )
                start_time = time.time()

    if steps_in_accum > 0:
        _optimizer_step()

    if skipped_batches:
        logging.info(
            f"Epoch {epoch:02d} skipped {skipped_batches} batches "
            f"due to insufficient subsampled frames."
        )

    return global_step, loss_meter.avg


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            features, input_lengths, targets, target_lengths, _ = batch
            features = features.to(device)
            input_lengths = input_lengths.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            with autocast(enabled=amp_enabled):
                logits, logit_lengths = model(features, input_lengths)
                if torch.any(target_lengths > logit_lengths):
                    continue
                loss = criterion(
                    logits.transpose(0, 1),
                    targets,
                    logit_lengths,
                    target_lengths,
                )

            loss_meter.update(loss.item(), n=features.size(0))

    return loss_meter.avg


def format_hours(hours: float) -> str:
    if hours and hours > 0:
        return f"~{hours:.2f}h"
    return "n/a"


# ------------------------------------------------------------
# main training entry
# ------------------------------------------------------------

def run_training(cfg: Dict) -> Dict:
    device = torch.device(cfg["device"])

    # tokenizer
    tokenizer_cfg = dict(cfg["tokenizer"])
    tokenizer_cfg.setdefault("lowercase", cfg["data"].get("text_lowercase", True))
    tokenizer = TextTokenizer(tokenizer_cfg)

    # feature extractors
    feature_kwargs = dict(cfg["feature_extractor"])
    train_extractor = LogMelFeatureExtractor(
        sample_rate=cfg["data"]["sample_rate"], **feature_kwargs
    )
    valid_extractor = LogMelFeatureExtractor(
        sample_rate=cfg["data"]["sample_rate"], **feature_kwargs
    )

    # datasets
    train_dataset = build_dataset(
        cfg,
        tokenizer,
        train_extractor,
        split=cfg["data"]["hf_dataset"].get("train_split", "train"),
    )
    valid_split = cfg["data"]["hf_dataset"].get("valid_split", "dev")
    valid_dataset = build_dataset(
        cfg,
        tokenizer,
        valid_extractor,
        split=valid_split,
    )

    # collator / dataloader
    subsampling_factor = max(1, cfg["model"].get("subsampling_factor", 1))
    min_subsample_len_multiplier = cfg["model"].get(
        "min_subsample_len_multiplier", 1
    )
    collate_fn = SpeechDataCollator(
        pad_to_multiple_of=subsampling_factor,
        subsampling_factor=subsampling_factor,
        min_subsample_len_multiplier=min_subsample_len_multiplier,
    )

    train_loader = build_dataloader(
        train_dataset,
        collate_fn,
        cfg["dataloader"],
        shuffle=cfg["dataloader"].get("shuffle", True),
    )
    valid_loader = build_dataloader(
        valid_dataset,
        collate_fn,
        cfg["dataloader"],
        shuffle=False,
    )

    # logging dataset info
    hours_train = format_hours(getattr(train_dataset, "total_hours", 0.0))
    hours_valid = format_hours(getattr(valid_dataset, "total_hours", 0.0))
    frame_ms = (
        cfg["feature_extractor"].get("hop_length", 160)
        / cfg["data"]["sample_rate"]
        * 1000
    )
    effective_stride = frame_ms * subsampling_factor
    logging.info(
        f"Train set: {len(train_dataset)} utterances ({hours_train}), "
        f"Valid set: {len(valid_dataset)} utterances ({hours_valid})"
    )
    logging.info(
        f"Subsampling factor {subsampling_factor} "
        f"⇒ encoder frame rate ≈ {effective_stride:.1f} ms"
    )

    # model
    num_classes = tokenizer.vocab_size
    model = Conformer(
        num_classes=num_classes,
        input_dim=cfg["model"]["input_dim"],
        encoder_dim=cfg["model"]["encoder_dim"],
        num_encoder_layers=cfg["model"]["num_layers"],
        num_attention_heads=cfg["model"]["num_attention_heads"],
        feed_forward_expansion_factor=cfg["model"]["feed_forward_expansion_factor"],
        conv_expansion_factor=cfg["model"]["conv_expansion_factor"],
        conv_kernel_size=cfg["model"]["conv_kernel_size"],
        input_dropout_p=cfg["model"]["dropout"],
        feed_forward_dropout_p=cfg["model"]["dropout"],
        attention_dropout_p=cfg["model"]["dropout"],
        conv_dropout_p=cfg["model"]["dropout"],
    ).to(device)

    logging.info(f"Conformer parameters: {model.count_parameters():,}")

    # optim / scheduler / amp
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optim"]["peak_lr"],
        betas=cfg["optim"]["betas"],
        eps=cfg["optim"]["eps"],
        weight_decay=cfg["optim"]["weight_decay"],
    )
    scheduler = WarmupExponentialDecayScheduler(
        optimizer,
        warmup_steps=cfg["scheduler"]["warmup_steps"],
        total_steps=cfg["scheduler"]["total_steps"],
        final_lr_scale=cfg["scheduler"]["final_lr_scale"],
    )

    amp_enabled = bool(cfg["trainer"]["use_amp"] and torch.cuda.is_available())
    scaler = GradScaler(enabled=amp_enabled)

    # resume 설정
    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    best_path: Optional[Path] = None

    resume_path = cfg["trainer"].get("resume_from")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and amp_enabled and ckpt["scaler_state"] is not None:
            scaler.load_state_dict(ckpt["scaler_state"])

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        best_val = ckpt.get("best_val", best_val)
        best_path = Path(resume_path)
        logging.info(f"Resumed from {resume_path} (epoch {ckpt['epoch']})")

    # training loop
    for epoch in range(start_epoch, cfg["trainer"]["num_epochs"] + 1):
        global_step, train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            grad_accum_steps=cfg["optim"]["grad_accum_steps"],
            grad_clip=cfg["trainer"]["grad_clip"],
            log_interval=cfg["trainer"]["log_interval"],
            amp_enabled=amp_enabled,
        )

        if epoch % cfg["trainer"]["val_interval"] == 0:
            val_loss = evaluate(
                model=model,
                dataloader=valid_loader,
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss

            ckpt_state = {
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": float(val_loss),
                "best_val": float(best_val),
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if amp_enabled else None,
                "config": cfg,
            }
            ckpt_path = save_checkpoint(
                ckpt_state,
                cfg["trainer"]["checkpoint_dir"],
                cfg["trainer"]["max_to_keep"],
            )
            if improved:
                best_path = ckpt_path

            logging.info(
                f"Epoch {epoch:02d} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | best {best_val:.4f}"
            )
        else:
            logging.info(f"Epoch {epoch:02d} | train loss {train_loss:.4f}")

    return {
        "best_val_loss": best_val,
        "best_checkpoint": str(best_path) if best_path else None,
        "global_step": global_step,
    }


# ------------------------------------------------------------
# entrypoint
# ------------------------------------------------------------

if __name__ == "__main__":
    set_seed(config["seed"])
    Path(config["trainer"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    logging.info(f"Running Stage 1 on device: {config['device']}")

    # GPU 선택은 셸에서:
    #   CUDA_VISIBLE_DEVICES=4,5,6,7 python train_stage1_ctc.py
    try:
        trainer_state = run_training(config)
        print(trainer_state)
    except Exception as exc:
        logging.exception("Training loop aborted.")
        print(f"Training loop aborted: {exc}")
