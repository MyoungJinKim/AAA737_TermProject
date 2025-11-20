# /data_x/aa007878/deep/myung/conformer/train/tokenizer.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class TextTokenizer:
    """
    Simple text tokenizer supporting:
      - character-level tokenization
      - SentencePiece-based tokenization

    cfg 예시:
    {
        "blank_id": 0,
        "lowercase": True,
        "use_sentencepiece": True,
        "sentencepiece_model": "/path/to/model.spm",
        "chars": "abcdefghijklmnopqrstuvwxyz' "
    }
    """

    def __init__(self, cfg: Dict):
        self.blank_id = int(cfg.get("blank_id", 0))
        self.lowercase = bool(cfg.get("lowercase", True))
        self.use_sentencepiece = bool(cfg.get("use_sentencepiece", False))

        sp_model = cfg.get("sentencepiece_model")

        if self.use_sentencepiece:
            if spm is None:
                raise ImportError(
                    "sentencepiece is required for the requested tokenizer "
                    "but is not installed."
                )

            if not sp_model or not Path(sp_model).is_file():
                raise FileNotFoundError(f"SentencePiece model not found: {sp_model}")

            self.processor = spm.SentencePieceProcessor(model_file=str(sp_model))
            # blank_id를 0으로 둘 경우, SentencePiece id 전체에 offset=1을 줘서 0을 blank로 사용
            self.offset = 1 if self.blank_id == 0 else 0
            self.vocab_size = self.processor.get_piece_size() + self.offset

        else:
            # char-level tokenizer
            base_chars = cfg.get("chars") or list("abcdefghijklmnopqrstuvwxyz' ")
            symbols: List[str] = []

            if self.blank_id == 0:
                symbols.append("<blank>")

            symbols.extend(base_chars)

            self.stoi = {ch: idx for idx, ch in enumerate(symbols)}
            self.itos = {idx: ch for ch, idx in self.stoi.items()}
            self.vocab_size = len(symbols)

    def encode(self, text: str) -> List[int]:
        if text is None:
            return []

        text = text.strip()
        if self.lowercase:
            text = text.lower()

        # SentencePiece 모드
        if hasattr(self, "processor"):
            ids = self.processor.encode(text, out_type=int)
            return [idx + self.offset for idx in ids]

        # char-level 모드
        tokens: List[int] = []
        for ch in text:
            idx = self.stoi.get(ch)
            # blank_id는 제외
            if idx is not None and idx != self.blank_id:
                tokens.append(idx)
        return tokens

    def decode(self, ids: List[int]) -> str:
        # SentencePiece 모드
        if hasattr(self, "processor"):
            shifted = [idx - self.offset for idx in ids if idx >= self.offset]
            return self.processor.decode(shifted)

        # char-level 모드
        return "".join(self.itos.get(idx, "") for idx in ids if idx != self.blank_id)
