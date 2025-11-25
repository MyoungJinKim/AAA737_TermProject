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
