# AAA737_TermProject

## 프로젝트 개요
이 프로젝트는 오디오 데이터를 처리하고 LLaMA 모델과 결합하여 학습하는 파이프라인을 구축합니다.

## 실행 방법 (How to Run)

`train.py` 스크립트를 사용하여 모델 학습을 시작할 수 있습니다.

### 1. 환경 설정
필요한 라이브러리가 설치되어 있는지 확인하세요. (예: `torch`, `numpy`, `yaml` 등)

### 2. 설정 파일 확인
학습 설정은 `configs/config.yaml` 파일에 정의되어 있습니다.
`train.py` 내부에서 해당 경로(`/data_x/aa007878/deep/myung/configs/config.yaml`)를 참조하고 있으므로, 필요에 따라 경로를 수정하거나 설정 파일 내용을 변경하세요.

### 3. 학습 실행

#### 단일 GPU 실행 (Single GPU)
단순히 코드를 테스트하거나 하나의 GPU만 사용하려면 아래와 같이 실행합니다. (이 경우 `train.py` 내부에서 분산 모드가 비활성화됩니다.)
```bash
python train.py
```

#### 멀티 GPU 실행 (Multi-GPU / DDP)
`train.py`에 작성된 분산 학습 코드(DDP)를 활성화하여 4개의 GPU를 모두 사용하려면 `torchrun`을 사용해야 합니다.
```bash
# GPU 4개를 사용하는 경우 (--nproc_per_node=4)
torchrun --nproc_per_node=4 train.py
```
*참고: `train.py` 코드 상단에 `os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"` 설정이 있으므로, 4개의 GPU가 있는 환경에서 위 명령어를 사용하면 됩니다.*

스크립트는 기본적으로 GPU 0, 1, 2, 3번을 사용하도록 설정되어 있습니다 (`os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"`).

### 4. 추론 실행 (Inference)
학습된 모델 체크포인트를 사용하여 오디오 파일을 텍스트로 변환(추론)할 수 있습니다.

```bash
python inference.py \
  --config configs/config.yaml \
  --checkpoint /data_x/aa007878/deep/myung/model/model_storage/stage2_best_lossX.XXXX.pth \
  --audio_path /path/to/your/audio.wav
```

*   `--config`: 학습에 사용한 설정 파일 경로 (기본값: `configs/config.yaml`)
*   `--checkpoint`: 학습된 모델 체크포인트 파일 경로 (`.pth`)
    *   학습 중 `model/model_storage/` 폴더에 저장된 파일을 사용하세요.
*   `--audio_path`: 변환할 오디오 파일 경로 (`.wav`)

### 5. 평가 실행 (Evaluation)
학습된 모델의 성능을 테스트 데이터셋으로 평가하고 WER(Word Error Rate)을 계산할 수 있습니다.

```bash
python evaluation.py \
  --config configs/evaluate_config.yaml \
  --ckpt {체크포인트 경로}
```

*   `--config`: 평가 설정 파일 경로 (기본값: `configs/evaluate_config.yaml`)
    *   평가에 사용할 데이터셋, 생성 파라미터 등을 설정합니다.
*   `--ckpt`: 평가할 모델 체크포인트 파일 경로 (`.pth`, 선택사항)
    *   지정하지 않으면 설정 파일의 `run.eval_checkpoint` 값을 사용합니다.
    *   학습 중 `model/model_storage/` 폴더에 저장된 파일을 사용하세요.

**평가 결과:**
*   터미널에는 최종 평균 WER만 출력됩니다.
*   상세한 평가 결과(각 샘플별 정답, 모델 출력, WER)는 `outputs/evaluation_{체크포인트명}_{타임스탬프}.txt` 파일에 저장됩니다.
*   결과 파일에는 개별 샘플 결과와 전체 평균 WER이 모두 포함됩니다.

## 코드 설명 (Code Description)

### `train.py`
모델 학습의 진입점(entry point) 역할을 하는 스크립트입니다. 주요 기능은 다음과 같습니다:

1.  **설정 로드 (`load_config`)**:
    *   YAML 설정 파일을 읽어와 학습 파라미터(배치 크기, 학습률, 모델 구조 등)를 로드합니다.
    *   GPU 사용 가능 여부에 따라 `device` 설정을 자동으로 조정합니다.

2.  **시드 설정 (`setup_seeds`)**:
    *   재현 가능한 결과를 위해 Random, NumPy, PyTorch의 시드(Seed)를 고정합니다.

3.  **분산 학습 초기화 (`init_distributed_mode`)**:
    *   멀티 GPU 환경에서의 학습을 위해 분산 학습 모드를 초기화합니다.

4.  **모델 및 데이터 로더 준비**:
    *   `load_model`: 설정에 맞춰 모델(Conformer + LLaMA Projection 등)을 불러옵니다.
    *   `build_dataset`, `build_dataloader`: 학습 및 검증 데이터를 로드하고 배치를 생성하는 데이터 로더를 구축합니다.

5.  **학습 실행 (`Runner`)**:
    *   `Runner` 클래스를 초기화하고 `runner.train()`을 호출하여 실제 학습 루프를 시작합니다.

### `evaluation.py`
모델 평가를 수행하는 스크립트입니다. 주요 기능은 다음과 같습니다:

1.  **설정 및 체크포인트 로드**:
    *   평가 설정 파일(`configs/evaluate_config.yaml`)을 읽어와 평가 파라미터를 로드합니다.
    *   학습된 모델 체크포인트를 로드하여 평가에 사용합니다.

2.  **데이터셋 준비**:
    *   설정 파일에 지정된 테스트/검증 데이터셋을 로드합니다.
    *   HuggingFace 데이터셋을 사용하여 평가 데이터를 준비합니다.

3.  **모델 추론 및 WER 계산**:
    *   각 샘플에 대해 모델이 생성한 텍스트와 정답 텍스트를 비교합니다.
    *   HuggingFace `evaluate` 라이브러리를 사용하여 WER(Word Error Rate)을 계산합니다.

4.  **결과 저장**:
    *   개별 샘플별 상세 결과(정답, 모델 출력, WER)를 파일로 저장합니다.
    *   터미널에는 최종 평균 WER만 출력하여 가독성을 높입니다.
