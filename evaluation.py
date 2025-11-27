import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO
from datetime import datetime

import torch
import yaml
from evaluate import load as load_metric
import soundfile as sf
import numpy as np

# torchcodec 에러를 완전히 피하기 위해 torchaudio.load를 soundfile로 대체
# 환경 변수 설정으로 torchcodec 로딩 시도 억제
os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "1")


# torchaudio import 전에 모듈을 패치하기 위해 먼저 설치
def install_safe_audio_loader():
    """
    TorchAudio의 torchcodec 의존성 문제를 완전히 피하기 위해,
    torchaudio.load를 soundfile 기반 로더로 완전히 대체한다.
    """

    def soundfile_load(fileobj, *args, **kwargs):
        """
        torchaudio.load를 soundfile로 완전히 대체.
        fileobj는 파일 경로(str/Path) 또는 file-like object일 수 있음.
        """
        # 파일 경로인 경우
        if isinstance(fileobj, (str, Path)):
            data, sr = sf.read(str(fileobj), dtype="float32", always_2d=True)
            waveform = torch.from_numpy(data.T)
            return waveform, sr

        # file-like object인 경우
        if hasattr(fileobj, "seek"):
            fileobj.seek(0)

        # BytesIO나 file-like object 처리
        if hasattr(fileobj, "read"):
            data, sr = sf.read(fileobj, dtype="float32", always_2d=True)
        else:
            # bytes인 경우
            if isinstance(fileobj, bytes):
                fileobj = BytesIO(fileobj)
            data, sr = sf.read(fileobj, dtype="float32", always_2d=True)

        waveform = torch.from_numpy(data.T)
        return waveform, float(sr)

    # stderr를 완전히 억제하여 torchcodec 에러 메시지 차단
    from io import StringIO
    import contextlib

    @contextlib.contextmanager
    def suppress_stderr():
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            yield
        finally:
            sys.stderr = old_stderr

    # torchaudio 모듈이 이미 로드되었는지 확인
    if "torchaudio" in sys.modules:
        torchaudio = sys.modules["torchaudio"]
        torchaudio.load = soundfile_load
        print("[INFO] torchaudio.load를 soundfile 기반 로더로 대체했습니다.")
    else:
        # 모듈이 아직 로드되지 않았으면, 로드 시 패치
        with suppress_stderr():
            try:
                import torchaudio

                torchaudio.load = soundfile_load
                print("[INFO] torchaudio.load를 soundfile 기반 로더로 대체했습니다.")
            except Exception:
                # import 실패해도 괜찮음 (데이터셋에서 처리)
                pass


# 데이터셋 import 전에 먼저 패치 설치
install_safe_audio_loader()

# 데이터셋 모듈이 torchaudio를 import할 때도 에러를 억제하기 위해
# import hook을 사용하여 패치 (더 안전한 방법)
import builtins

_original_import = builtins.__import__


def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torchaudio":
        # torchaudio import 시 stderr 완전히 억제
        from io import StringIO

        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            module = _original_import(name, globals, locals, fromlist, level)
            # import 후 load 함수 패치
            if hasattr(module, "load"):

                def soundfile_load(fileobj, *args, **kwargs):
                    if isinstance(fileobj, (str, Path)):
                        data, sr = sf.read(
                            str(fileobj), dtype="float32", always_2d=True
                        )
                        waveform = torch.from_numpy(data.T)
                        return waveform, sr
                    if hasattr(fileobj, "seek"):
                        fileobj.seek(0)
                    if hasattr(fileobj, "read"):
                        data, sr = sf.read(fileobj, dtype="float32", always_2d=True)
                    else:
                        if isinstance(fileobj, bytes):
                            fileobj = BytesIO(fileobj)
                        data, sr = sf.read(fileobj, dtype="float32", always_2d=True)
                    waveform = torch.from_numpy(data.T)
                    return waveform, float(sr)

                module.load = soundfile_load
            return module
        finally:
            sys.stderr = old_stderr
    return _original_import(name, globals, locals, fromlist, level)


builtins.__import__ = _patched_import

from data.dataloader import build_dataset, build_dataloader
from model import load_model
from utils import prepare_sample, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute WER using HuggingFace evaluate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluate_config.yaml"),
        help="학습/데이터 설정 YAML 경로",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="stage2 checkpoint(.pth) 경로 (미입력 시 config의 eval_checkpoint 사용)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Tuple[Dict, Path]:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"

    return cfg, path.resolve().parent


def resolve_path(path_str: str, base_dir: Path) -> Path:
    candidate = Path(path_str)
    return candidate if candidate.is_absolute() else base_dir / candidate


def resolve_model_paths(model_cfg: Dict, base_dir: Path) -> Dict:
    resolved = model_cfg.copy()
    for key in ("conformer_path", "speech_llama_proj_model"):
        value = resolved.get(key)
        if value:
            resolved[key] = str(resolve_path(value, base_dir))
    return resolved


def resolve_run_paths(run_cfg: Dict, base_dir: Path) -> Dict:
    resolved = run_cfg.copy()
    if "model_storage_path" in resolved and resolved["model_storage_path"]:
        resolved["model_storage_path"] = str(
            resolve_path(resolved["model_storage_path"], base_dir)
        )
    if "eval_checkpoint" in resolved and resolved["eval_checkpoint"]:
        resolved["eval_checkpoint"] = resolve_path(
            resolved["eval_checkpoint"], base_dir
        )
    return resolved


def is_lfs_pointer(ckpt_path: Path) -> bool:
    """
    Detect a Git LFS pointer file without loading large checkpoints into memory.
    """
    try:
        size = ckpt_path.stat().st_size
        if size > 2048:
            return False
        with ckpt_path.open("rb") as f:
            head = f.read(2048)
    except OSError:
        return False

    marker = b"https://git-lfs.github.com/spec/v1"
    return marker in head


def ensure_checkpoint_ready(ckpt_path: Path) -> None:
    if is_lfs_pointer(ckpt_path):
        raise RuntimeError(
            f"Checkpoint looks like a Git LFS pointer: {ckpt_path}\n"
            "실제 가중치가 없어서 torch.load에 실패합니다. "
            "`git lfs install && git lfs pull` 후 다시 시도하세요."
        )


def get_checkpoint_path(arg_ckpt: Optional[Path], run_cfg: Dict) -> Optional[Path]:
    if arg_ckpt:
        return arg_ckpt
    ckpt_from_cfg = run_cfg.get("eval_checkpoint")
    if ckpt_from_cfg:
        return Path(ckpt_from_cfg)
    return None


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ensure_checkpoint_ready(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    loaded_state = checkpoint.get("model", checkpoint)

    # LoRA 학습 시 저장된 state_dict에는 베이스 LLaMA 가중치가 제외될 수 있음
    # (requires_grad=False 항목이 제거됨). 현재 모델의 state_dict을 기반으로
    # 로드 가능한 키만 덮어써서 불필요한 missing 경고를 없앤다.
    model_state = model.state_dict()
    unused_keys = []
    for k, v in loaded_state.items():
        if k in model_state:
            model_state[k] = v
        else:
            unused_keys.append(k)

    model.load_state_dict(model_state, strict=False)
    if unused_keys:
        print(
            f"[INFO] Skipped {len(unused_keys)} keys not present in current model (e.g., {unused_keys[:3]})"
        )


def get_eval_split(cfg: Dict) -> str:
    hf_cfg = cfg.get("data", {}).get("hf_dataset", {})
    for key in (
        "test_split",
        "valid_split",
        "validation_split",
        "dev_split",
        "train_split",
    ):
        split = hf_cfg.get(key)
        if split:
            return split
    return "test"


def build_generate_cfg(cfg: Dict) -> Dict:
    defaults = {
        "max_new_tokens": 128,
        "num_beams": 4,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
    }
    allowed_keys = set(defaults.keys())
    source_cfg = cfg.get("generate")
    if source_cfg is None:
        source_cfg = cfg.get("run", {})

    for k, v in source_cfg.items():
        if k in allowed_keys:
            defaults[k] = v
    return defaults


def main():
    args = parse_args()
    setup_logger()
    # torchaudio 패치는 이미 모듈 import 전에 수행됨

    cfg, cfg_dir = load_config(args.config)
    repo_dir = (
        cfg_dir.parent
    )  # config 파일이 configs/ 아래 있을 때 프로젝트 루트 기준으로 경로 해석
    cfg["run"] = resolve_run_paths(cfg.get("run", {}), repo_dir)
    cfg["model"] = resolve_model_paths(cfg.get("model", {}), repo_dir)
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg["device"] = str(device)

    model = load_model(cfg["model"])
    model.to(device)
    model.device = device
    # 평가 시 dtype 불일치 방지를 위해 모델을 float32로 변환
    # (모델이 float16으로 로드되었을 수 있지만, 입력은 float32)
    model = model.float()
    model.eval()

    ckpt_path = get_checkpoint_path(args.ckpt, cfg.get("run", {}))
    if ckpt_path:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"[INFO] Loading checkpoint from {ckpt_path}")
        load_checkpoint(model, ckpt_path)
    else:
        print("[WARN] No checkpoint provided; evaluating with initial weights.")

    eval_split = get_eval_split(cfg)
    hf_name = cfg.get("data", {}).get("hf_dataset", {}).get("name", "UNKNOWN")
    print(f"[INFO] Loading HF dataset '{hf_name}' split '{eval_split}' for evaluation")
    dataset = build_dataset(cfg, split=eval_split)
    dataloader = build_dataloader(dataset, cfg.get("dataloader", {}), shuffle=False)

    generate_cfg = build_generate_cfg(cfg)
    wer_metric = load_metric("wer")
    use_cuda = device.type == "cuda"

    evaluated = 0
    skipped_zero_ref = 0
    references_all: List[str] = []
    predictions_all: List[str] = []
    sample_results: List[Dict[str, Any]] = []

    # 결과 파일 경로 설정
    output_dir = Path(cfg.get("run", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = ckpt_path.stem if ckpt_path else "no_ckpt"
    result_file = output_dir / f"evaluation_{ckpt_name}_{timestamp}.txt"

    print(f"[INFO] 평가 결과를 파일에 저장합니다: {result_file}")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("개별 샘플 평가 결과\n")
        f.write("=" * 80 + "\n\n")

        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                references = batch["text"]

                samples = prepare_sample(batch, cuda_enabled=use_cuda)
                predictions = model.generate(samples, generate_cfg)

                for sample_idx, (ref, hyp) in enumerate(zip(references, predictions)):
                    if not ref or not ref.strip():
                        skipped_zero_ref += 1
                        continue

                    # 개별 샘플 WER 계산
                    sample_wer = wer_metric.compute(references=[ref], predictions=[hyp])

                    # 샘플 정보를 파일에 저장
                    global_idx = evaluated + 1
                    f.write(f"\n[Sample {global_idx}]\n")
                    f.write(f"  정답 (Reference): {ref}\n")
                    f.write(f"  모델 출력 (Prediction): {hyp}\n")
                    f.write(f"  WER: {sample_wer * 100:.2f}%\n")
                    f.write("-" * 80 + "\n")

                    # 결과 저장
                    sample_results.append(
                        {
                            "index": global_idx,
                            "reference": ref,
                            "prediction": hyp,
                            "wer": sample_wer * 100,
                        }
                    )

                    references_all.append(ref)
                    predictions_all.append(hyp)
                    evaluated += 1

        if not references_all:
            error_msg = "평가할 참조 텍스트가 없습니다. (빈 reference만 존재)"
            f.write(error_msg + "\n")
            print(error_msg)
            return

        # 전체 WER 계산
        wer = wer_metric.compute(references=references_all, predictions=predictions_all)

        # 최종 결과를 파일에 저장
        f.write("\n" + "=" * 80 + "\n")
        f.write("전체 평가 결과\n")
        f.write("=" * 80 + "\n")
        f.write(f"Evaluated samples : {evaluated}\n")
        f.write(f"Word Error Rate   : {wer * 100:.2f}%\n")
        if skipped_zero_ref:
            f.write(f"Skipped samples with empty reference: {skipped_zero_ref}\n")
        f.write("=" * 80 + "\n")

    # 최종 평균 WER만 터미널에 출력
    print("\n" + "=" * 80)
    print("전체 평가 결과")
    print("=" * 80)
    print(f"Evaluated samples : {evaluated}")
    print(f"Word Error Rate   : {wer * 100:.2f}%")
    if skipped_zero_ref:
        print(f"Skipped samples with empty reference: {skipped_zero_ref}")
    print(f"결과 파일: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
