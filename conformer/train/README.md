# Conformer Stage 1 CTC 학습 (Train Stage 1 CTC)

이 디렉토리는 Conformer 모델의 인코더(Encoder)를 CTC(Connectionist Temporal Classification) Loss를 사용하여 사전 학습(Pre-training)하기 위한 스크립트를 포함하고 있습니다.

## 파일 설명

- `train_stage1_ctc.py`: Conformer 모델의 Stage 1 학습을 수행하는 메인 스크립트입니다. PyTorch DDP(Distributed Data Parallel)를 지원하며, WANDB 로깅 및 체크포인트 저장을 수행합니다.

## 실행 방법

이 스크립트는 `torchrun`을 사용하여 단일 노드 멀티 GPU 환경에서 실행하도록 구성되어 있습니다.

### 1. 환경 변수 설정 (선택 사항)

실행 전 사용할 GPU 및 기타 설정을 환경 변수로 지정할 수 있습니다.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 사용할 GPU ID
export TOKENIZERS_PARALLELISM=false  # 토크나이저 병렬 처리 경고 방지
export NCCL_P2P_DISABLE=1            # NCCL P2P 통신 문제 발생 시 설정
```

### 2. 학습 실행

프로젝트 루트 디렉토리(예: `/data_x/aa007878/deep/myung`) 또는 이 폴더 내에서 아래 명령어를 실행합니다.

```bash
torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_stage1_ctc.py
```

- `--nproc_per_node`: 사용할 GPU 개수 (위 예시는 4개)

## 설정 (Configuration)

`train_stage1_ctc.py` 파일 내부의 `config` 딕셔너리를 수정하여 학습 파라미터를 변경할 수 있습니다.

### 주요 설정 항목

- **Data (`data`)**:
  - `hf_dataset`: HuggingFace 데이터셋 설정 (기본: `parler-tts/mls_eng`)
  - `sample_rate`: 오디오 샘플링 레이트 (기본: 16000)
  - `max_audio_seconds`: 최대 오디오 길이 (초)

- **Model (`model`)**:
  - `encoder_dim`: 인코더 차원 (기본: 512)
  - `num_layers`: 인코더 레이어 수 (기본: 12)
  - `num_attention_heads`: 어텐션 헤드 수 (기본: 8)

- **Dataloader (`dataloader`)**:
  - `batch_size`: GPU당 배치 크기 (기본: 32)
  - `num_workers`: 데이터 로딩 워커 수

- **Optimizer & Scheduler (`optim`, `scheduler`)**:
  - `peak_lr`: 최대 학습률 (기본: 1e-3)
  - `warmup_steps`: 웜업 스텝 수 (기본: 30000)

- **Trainer (`trainer`)**:
  - `num_epochs`: 총 학습 에폭 수
  - `val_steps`: 검증(Validation) 수행 간격 (Step 단위, 기본: 1000)
  - `ckpt_steps`: 체크포인트 저장 간격 (Step 단위, 기본: 5000)
  - `checkpoint_dir`: 체크포인트 저장 경로
  - `use_amp`: 자동 혼합 정밀도(AMP) 사용 여부

- **WandB (`wandb`)**:
  - `enable`: WANDB 로깅 사용 여부
  - `project`: WANDB 프로젝트 이름
  - `run_name`: 실행 이름 (None일 경우 자동 생성)

## 결과물 (Outputs)

학습이 진행되면 `config['trainer']['checkpoint_dir']`에 지정된 경로(기본: `checkpoints/stage1_ctc_layer12_batch32_2`)에 다음과 같은 파일들이 저장됩니다.

- `epoch{XX}_val{LOSS}.pt`: 각 시점의 모델 체크포인트 (모델 가중치, 옵티마이저 상태 등 포함)
- `best_checkpoint.txt`: 가장 성능이 좋은(Validation Loss가 낮은) 체크포인트 파일명 기록

## 의존성 (Dependencies)

이 스크립트는 상위 디렉토리의 다음 모듈들을 사용합니다.
- `tokenizer`: 텍스트 토크나이저
- `data`: 데이터셋 로딩 및 전처리 (`features`, `collate`, `dataloader`)
- `conformer`: Conformer 모델 정의

실행 시 `sys.path` 설정을 통해 상위 디렉토리를 자동으로 참조합니다.
