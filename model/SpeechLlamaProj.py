import logging
import torch
import torch.nn as nn
from typing import Optional


class SpeechLlamaProj(nn.Module):
    def __init__(
        self,
        in_dim: int,                       # conformer_dim * conformer_concat_num
        out_dim: int,                      # llama hidden size
        pretrained_path: Optional[str] = None,
        freeze: bool = False,
        key_in_ckpt: str = "speech_llama_proj",  # 체크포인트 안에서의 키 이름
    ):
        super().__init__()

        # 실제 projection layer
        self.proj = nn.Linear(in_dim, out_dim)

        # 옵션 저장
        self.pretrained_path = pretrained_path
        self.freeze = freeze
        self.key_in_ckpt = key_in_ckpt

        # 초기화 후 바로 로드/프리즈 처리
        self._maybe_load_pretrained()
        self._maybe_freeze()

    def _maybe_load_pretrained(self):
        """pretrained checkpoint가 있을 때만, shape 맞으면 load."""
        if not self.pretrained_path:
            logging.info("[SpeechLlamaProj] No pretrained_path provided. Using fresh init.")
            return

        logging.info(f"[SpeechLlamaProj] Loading from {self.pretrained_path}")
        ckpt = torch.load(self.pretrained_path, map_location="cpu")

        # 1) ckpt 안에 'speech_llama_proj'만 따로 저장된 경우
        if self.key_in_ckpt in ckpt:
            proj_state = ckpt[self.key_in_ckpt]
        # 2) 통째로 저장해놨는데 그 안에 proj의 키만 골라야 하는 경우
        elif "model" in ckpt and self.key_in_ckpt in ckpt["model"]:
            proj_state = ckpt["model"][self.key_in_ckpt]
        else:
            # 일단 전체 state_dict를 가져와보고, proj의 키만 필터링해서 사용
            state_dict = ckpt.get("model", ckpt)
            proj_state = {
                k.replace("speech_llama_proj.", ""): v
                for k, v in state_dict.items()
                if k.startswith("speech_llama_proj.")
            }

        # weight shape가 현재 proj와 맞는지 확인
        w = proj_state.get("weight", None)
        b = proj_state.get("bias", None)
        if w is None:
            logging.warning("[SpeechLlamaProj] No 'weight' in loaded state. Skip loading.")
            return

        if w.shape != self.proj.weight.shape or (b is not None and b.shape != self.proj.bias.shape):
            logging.warning(
                f"[SpeechLlamaProj] Shape mismatch. "
                f"ckpt weight {tuple(w.shape)} vs current {tuple(self.proj.weight.shape)}. "
                "Using fresh initialization instead."
            )
            return

        missing, unexpected = self.proj.load_state_dict(proj_state, strict=False)
        logging.info(
            f"[SpeechLlamaProj] Loaded pretrained projector. "
            f"missing={missing}, unexpected={unexpected}"
        )

    def _maybe_freeze(self):
        """freeze=True인 경우 학습에서 제외."""
        if not self.freeze:
            return

        for name, param in self.proj.named_parameters():
            param.requires_grad = False
        self.proj.eval()
        logging.info("[SpeechLlamaProj] Projection is frozen.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        return self.proj(x)
