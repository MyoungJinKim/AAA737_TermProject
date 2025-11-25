import argparse
import os
import yaml
import torch
import torchaudio
from pathlib import Path
from model import load_model
from utils import prepare_sample

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_checkpoint(model, checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. 모델 state_dict 가져오기
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # 2. 모델에 로드 (strict=False로 설정하여 부분 로드 허용)
    #    - LoRA, Projector 등 학습된 파라미터만 업데이트됨
    #    - 원본 LLaMA 등은 초기화 시점의 상태 유지 (Freeze된 상태라면)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"Checkpoint loaded.")
    if missing_keys:
        print(f" - Missing keys (expected for frozen parts): {len(missing_keys)} keys")
        # print(f"   Example: {missing_keys[:5]}")
    if unexpected_keys:
        print(f" - Unexpected keys: {unexpected_keys}")

    return model

def inference(args):
    # 1. 설정 로드
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)

    # 2. 모델 빌드 (기본 구조 초기화)
    print("Building model...")
    model_config = config.get("model", {})
    # 추론 시에는 학습 관련 설정 일부 무시 가능하지만, 구조는 동일해야 함
    model = load_model(model_config)
    model.to(device)
    model.eval()

    # 3. 체크포인트 로드 (학습된 가중치 덮어쓰기)
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint, device)
    else:
        print("Warning: No checkpoint provided. Using random/initial weights.")

    # 4. 오디오 전처리
    print(f"Processing audio: {args.audio_path}")
    waveform, sample_rate = torchaudio.load(args.audio_path)
    
    # 리샘플링 (모델이 16k를 기대한다고 가정)
    target_sr = config["data"].get("sample_rate", 16000)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Mel Spectrogram 변환 (feature_extractor 설정 사용)
    # 여기서는 간단히 torchaudio transforms 사용하거나, 
    # data/features.py의 로직을 가져와야 정확함.
    # 편의상 data.features 모듈이 있다고 가정하고 유사하게 구현하거나
    # 프로젝트 내의 feature extractor를 사용하는 것이 좋음.
    
    # 임시: data/features.py의 Processor 등을 사용하는 것이 정석이지만,
    # 여기서는 간단히 모델이 기대하는 입력 형태를 맞추는 예시를 작성
    # 실제로는 학습 때 사용한 FeatureExtractor를 써야 함.
    
    # (B, T, F) 형태로 변환 필요. 
    # 여기서는 간단히 0으로 채운 더미 데이터를 넣거나, 
    # 실제로는 data.dataloader.build_dataset 등을 통해 전처리 파이프라인을 태우는 게 좋음.
    
    # 하지만 사용자가 단일 파일 추론을 원하므로, 간단한 Mel 변환 로직을 구현하거나
    # 기존 코드를 재사용해야 함. 
    # 여기서는 `data.features` 모듈을 활용하는 방향으로 가정.
    
    from data.features import SpeechFeaturizer
    featurizer = SpeechFeaturizer(config["feature_extractor"])
    
    # featurizer는 (T,) waveform을 받아 (T_mel, n_mels) 텐서 반환 가정
    # 만약 waveform이 (1, T)라면 (T,)로 변경
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
        
    features = featurizer(waveform) # (T_mel, n_mels)
    features = features.unsqueeze(0).to(device) # (1, T_mel, n_mels)
    input_lengths = torch.tensor([features.size(1)], device=device) # (1,)
    
    # 5. 입력 구성
    samples = {
        "input_features": features,
        "input_input_lengths": input_lengths,
        "text": [""] # 텍스트 프롬프트가 필요하다면 여기에 입력 (예: "Transcribe this:")
    }

    # 6. 생성 (Generate)
    print("Generating...")
    generate_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": False,
        "temperature": 1.0,
    }
    
    with torch.no_grad():
        # model.generate 내부에서 autocast 등을 처리한다고 가정
        generated_text = model.generate(samples, generate_cfg)

    print("-" * 30)
    print(f"Output: {generated_text[0]}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input audio file (.wav)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    
    args = parser.parse_args()
    inference(args)
