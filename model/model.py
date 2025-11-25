from __future__ import annotations
import logging
import json
import contextlib
import random
from typing import Dict, Any

from .utils import StoppingCriteriaSub
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, StoppingCriteriaList, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model


# 기존: from conformer import Conformer  (❌ 현재 경로 구조와 안 맞음)
# 수정: 같은 패키지(model) 안의 서브패키지에서 상대 import
from .conformer.conformer.model import Conformer, ConformerEncoderOnly, load_encoder_from_checkpoint

from .SpeechLlamaProj import SpeechLlamaProj

class modelYIM(nn.Module):
    # -----------------------------
    # Conformer 초기화
    # -----------------------------
    def _init_conformer(
        self,
        input_dim: int,
        encoder_dim: int,
        num_encoder_layers: int,
        num_attention_heads: int,
        feed_forward_expansion_factor: int,
        conv_expansion_factor: int,
        conv_kernel_size: int,
        dropout: float,
        modelpath: str,
    ) -> ConformerEncoderOnly:
        """
        config["model"]의 설정에 맞춰 Conformer encoder를 초기화.
        필요하다면 modelpath(=checkpoint)를 내부에서 로드하는 Conformer 구현이라고 가정.
        """
        conformer = ConformerEncoderOnly(
            input_dim=input_dim,
            encoder_dim=encoder_dim,
            num_encoder_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            input_dropout_p=dropout,
            feed_forward_dropout_p=dropout,
            attention_dropout_p=dropout,
            conv_dropout_p=dropout,
        )
        load_encoder_from_checkpoint(
            checkpoint_path=modelpath, model=conformer)
        return conformer

    
    # -----------------------------
    # autocast 헬퍼
    # -----------------------------
    def maybe_autocast(self, dtype=torch.float16):
        """
        CPU면 autocast 비활성화, GPU면 지정 dtype으로 autocast 활성화.
        """
        enable_autocast = self.device.type == "cuda"
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        # ---- LLaMA / Conformer 경로 ----
        llama_path: str = "",
        conformer_path: str = "",

        # ---- Conformer 구조 ----
        conformer_dim: int = 512,
        conformer_layers: int = 12,
        conformer_input_dim: int = 80,
        conformer_concat_num: int = 3,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        subsampling_factor: int = 8,
        min_subsample_len_multiplier: int = 2,

        # ---- Speech → LLaMA projection ----
        speech_llama_proj_model: str = "",
        freeze_speech_llama_proj: bool = False,

        # ---- LoRA / LLaMA 학습 설정 ----
        lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        train_llama: bool = False,

        max_txt_len: int = 128,
    ):
        
        # config 값 보관 (필요 시 forward 등에서 사용)
        self.conformer_dim = conformer_dim
        self.conformer_layers = conformer_layers
        self.conformer_input_dim = conformer_input_dim
        self.conformer_concat_num = conformer_concat_num
        self.num_attention_heads = num_attention_heads
        self.feed_forward_expansion_factor = feed_forward_expansion_factor
        self.conv_expansion_factor = conv_expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout
        self.subsampling_factor = subsampling_factor
        self.min_subsample_len_multiplier = min_subsample_len_multiplier

        self.lora = lora
        self.train_llama = train_llama
        self.max_txt_len = max_txt_len

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora = lora
        self.train_llama = train_llama
        self.max_txt_len = max_txt_len

        if not llama_path:
            raise ValueError("The 'llama_path' in config.yaml is empty. Please specify the path to your LLaMA model (e.g., 'meta-llama/Llama-2-7b-hf' or a local path).")

        logging.info(f'Loading LLaMA Tokenizer from {llama_path}')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.float16,
        )
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        logging.info('Loading LLaMA Done')

        # 🔹 학습 모드 결정 로직
        if self.lora:
            # LoRA를 쓰는 경우: base LLaMA는 freeze, LoRA 모듈만 학습
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            logging.info('Base LLaMA is frozen. LoRA adapters will be trainable.')

            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training Enabled')

        else:
            # LoRA를 쓰지 않는 경우: train_llama flag로 전체 LLaMA 학습 여부 결정
            if not self.train_llama:
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
                logging.info('LLaMA is frozen (no LoRA, no full fine-tuning).')
            else:
                logging.info('Full LLaMA fine-tuning is enabled (no LoRA).')


        assert conformer_path
        logging.info('Loading Conformer Model')
        self.conformer = self._init_conformer(
            modelpath=conformer_path,
            input_dim=conformer_input_dim,
            encoder_dim=conformer_dim,
            num_encoder_layers=conformer_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        # Conformer 출력에 대해 layer norm 적용 (차원: conformer_dim)
        self.ln_speech = nn.LayerNorm(conformer_dim)

    
        logging.info('Loading speech LLAMA proj')
        logging.info("Initializing speech LLAMA proj")
        in_dim = conformer_dim * conformer_concat_num
        out_dim = self.llama_model.config.hidden_size

        self.speech_llama_proj = SpeechLlamaProj(
            in_dim=in_dim,
            out_dim=out_dim,
            pretrained_path=speech_llama_proj_model if speech_llama_proj_model else None,
            freeze=freeze_speech_llama_proj,
            key_in_ckpt="speech_llama_proj",         # ckpt 구조에 맞게 조정 가능
        )


    def forward(self, samples, verbose: bool = False):

        # ------------------------------------------------------------------
        # 1) 오디오 인코딩 (Conformer)
        #   - collator가 만들어 준 key 사용:
        #       "input_features":      [B, T_max, F]
        #       "input_input_lengths": [B]
        # ------------------------------------------------------------------
        """
            "input_features": feats,  # [T_i, F]
            "feature_length": feat_len,
            "text": text_value,
            "utt_id": utt_id,
        }
        """
        features = samples["input_features"]          # [B, T_max, F]
        input_lengths = samples["feature_length"]  # [B]
        texts = samples["text"]                      # List[str]

        # 디바이스 정렬 (prepare_sample에서 이미 옮겼다면 중복될 수 있음)
        features = features.to(self.device)
        input_lengths = input_lengths.to(self.device)

        # Conformer: (B, T_max, F) + length[B] → (B, T_enc, C_enc), out_lengths[B]
        # stage1에서 쓰던 시그니처: outputs, output_lengths = model(inputs, input_lengths)
        speech_embeds, out_lengths = self.conformer(features, input_lengths)
        # speech_embeds: [B, T_enc, C_enc]

        # 여기서 위의 speech embeds를 time 마다 conformer_concat_num만큼 concat한 뒤 LLaMA 차원으로 바꿔줘야 함

        B, T_enc, C_enc = speech_embeds.shape
        conformer_concat_num = getattr(self, "conformer_concat_num", 1)

        if conformer_concat_num > 1:
            # T_enc이 concat_num의 배수가 아닐 수 있으므로 뒤를 잘라서 맞춤
            T_trim = (T_enc // conformer_concat_num) * conformer_concat_num
            if T_trim != T_enc:
                speech_embeds = speech_embeds[:, :T_trim, :]
                out_lengths = out_lengths.clamp(max=T_trim)

                T_enc = T_trim

            T_new = T_enc // conformer_concat_num  # concat 후 토큰 개수
            # (B, T_enc, C_enc) → (B, T_new, C_enc * concat_num)
            concat_embeds = speech_embeds.view(
                B,
                T_new,
                C_enc * conformer_concat_num,
            )
            speech_token_lengths = (out_lengths // conformer_concat_num)
        else:
            # concat_num = 1 이면 그대로 사용
            concat_embeds = speech_embeds
            T_new = T_enc
            speech_token_lengths = out_lengths


        # 모든 프레임을 유효 토큰으로 사용 → attention mask = 1
        # (B, T_new)
        speech_embeds = self.speech_llama_proj(concat_embeds)
        max_T = speech_embeds.size(1)
        idx = torch.arange(max_T, device=self.device).unsqueeze(0)  # [1, T_new]
        speech_atts = (idx < speech_token_lengths.unsqueeze(1)).long()  # [B, T_new]

        # prepare inputs for LLM
        text_with_eos = [t + self.llama_tokenizer.eos_token for t in texts]

        to_regress_tokens = self.llama_tokenizer(
            text_with_eos,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False,
        ).to(self.device)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        # ------------------------------------------------------------------
        # 5) BOS + speech + text → LLaMA 입력 구성
        # ------------------------------------------------------------------
        B = speech_embeds.size(0)

        # BOS 토큰 id: (B, 1)
        bos_ids = torch.full(
            (B, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=to_regress_tokens.input_ids.dtype,
            device=self.device,
        )

        if not self.lora:
            bos_embeds = self.llama_model.model.embed_tokens(bos_ids)
        else:
            bos_embeds = self.llama_model.model.model.embed_tokens(bos_ids)

        # BOS 자리에 대응하는 attention mask: (B, 1)
        atts_bos = torch.ones(
            (B, 1),
            dtype=speech_atts.dtype,
            device=self.device,
        )

        # LLaMA용 input_embeds: [BOS] + [SPEECH] + [TEXT]
        # shape: (B, 1 + T_new + T_txt, hidden_llama)
        inputs_embeds = torch.cat(
            [bos_embeds, speech_embeds, to_regress_embeds],
            dim=1,
        )

        # attention mask: (B, 1 + T_new + T_txt)
        attention_mask = torch.cat(
            [atts_bos, speech_atts, to_regress_tokens.attention_mask],
            dim=1,
        )

        # 타깃: BOS + speech 구간은 loss를 계산하지 않도록 -100
        # empty_targets: (B, 1 + T_new)
        empty_targets = torch.full(
            (B, 1 + T_new),
            fill_value=-100,
            dtype=torch.long,
            device=self.device,
        )
        # 최종 targets: [bos+speech] = -100, 그 뒤 텍스트 타깃
        # shape: (B, 1 + T_new + T_txt)
        targets_full = torch.cat([empty_targets, targets], dim=1)

        # ------------------------------------------------------------------
        # 6) LLaMA forward + loss 계산
        # ------------------------------------------------------------------
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets_full,
            )
            loss = outputs.loss

        if verbose:
            # 디코딩 정확도 계산 (텍스트 구간만)
            nvocab = self.llama_model.config.vocab_size

            # logits에서 [bos+speech] 구간을 건너뛰고 텍스트 구간만 가져오기
            offset = empty_targets.size(1)  # bos+speech length
            # 텍스트 위치의 logits: [B, T_txt, V]
            text_logits = outputs.logits[:, offset:-1, :]  # shift 한 칸 고려 (마지막 토큰 예측은 실제 라벨 없음)
            pred_ids = text_logits.contiguous().view(-1, nvocab).argmax(dim=-1)

            labels = targets_full[:, offset:].contiguous().view(-1)
            mask = (labels != -100)
            correct = (pred_ids[mask] == labels[mask]).float().sum()
            total = mask.sum().item()

            return {"loss": loss, "correct": correct, "total": total}

        return outputs
    
    def generate(self, samples, generate_cfg):
        """
        samples:
        - "input_features":      [B, T_max, F]
        - "input_input_lengths": [B]

        generate_cfg: dict
        - max_new_tokens, num_beams, do_sample, min_length, temperature,
            top_p, repetition_penalty, length_penalty 등

        """

        # ------------------------------------------------------------------
        # 1) 오디오 인코딩 (Conformer) - forward와 동일한 입력 형식 사용
        # ------------------------------------------------------------------
        features = samples["input_features"].to(self.device)          # [B, T_max, F]
        input_lengths = samples["input_input_lengths"].to(self.device)  # [B]
        batch_size = features.size(0)

        # Conformer: (B, T_max, F) + length[B] → (B, T_enc, C_enc), out_lengths[B]
        speech_embeds, out_lengths = self.conformer(features, input_lengths)
        # speech_embeds: [B, T_enc, C_enc]

        B, T_enc, C_enc = speech_embeds.shape
        conformer_concat_num = getattr(self, "conformer_concat_num", 1)

        # ------------------------------------------------------------------
        # 2) time-axis concat (stacking) + 길이 반영 (forward와 동일 로직)
        # ------------------------------------------------------------------
        if conformer_concat_num > 1:
            # T_enc이 concat_num의 배수가 아닐 수 있으므로 뒤를 잘라서 맞춤
            T_trim = (T_enc // conformer_concat_num) * conformer_concat_num
            if T_trim != T_enc:
                speech_embeds = speech_embeds[:, :T_trim, :]
                out_lengths = out_lengths.clamp(max=T_trim)
                T_enc = T_trim

            T_new = T_enc // conformer_concat_num  # concat 후 토큰 개수

            # (B, T_enc, C_enc) → (B, T_new, C_enc * concat_num)
            concat_embeds = speech_embeds.view(
                B,
                T_new,
                C_enc * conformer_concat_num,
            )

            # 길이도 concat 이후 토큰 단위로 변환
            speech_token_lengths = (out_lengths // conformer_concat_num)  # [B]
        else:
            concat_embeds = speech_embeds
            T_new = T_enc
            speech_token_lengths = out_lengths  # [B]

        # ------------------------------------------------------------------
        # 3) Speech → LLaMA projection + speech attention mask
        # ------------------------------------------------------------------
        # (원하면 ln_speech 먼저 적용 가능)
        # concat_embeds = self.ln_speech(concat_embeds)

        # (B, T_new, C_enc * concat_num) → (B, T_new, hidden_llama)
        speech_embeds = self.speech_llama_proj(concat_embeds)  # [B, T_new, H_llama]

        # speech attention mask: padding 제외, 1=유효, 0=pad
        max_T = speech_embeds.size(1)
        idx = torch.arange(max_T, device=self.device).unsqueeze(0)       # [1, T_new]
        speech_atts = (idx < speech_token_lengths.unsqueeze(1)).long()   # [B, T_new]

        # ------------------------------------------------------------------
        # 4) BOS + (선택) 텍스트 프롬프트 + speech → LLaMA generate 입력 구성
        # ------------------------------------------------------------------
        # BOS 토큰 id 텐서: (B, 1)
        bos_ids = torch.full(
            (batch_size, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )

        if not self.lora:
            bos_embeds = self.llama_model.model.embed_tokens(bos_ids)  # [B, 1, H]
        else:
            bos_embeds = self.llama_model.model.model.embed_tokens(bos_ids)

        # BOS attention mask: (B,1)
        atts_bos = torch.ones(
            (batch_size, 1),
            dtype=torch.long,
            device=self.device,
        )

        # 기본: [BOS] + [SPEECH]
        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)  # [B, 1+T_new, H]
        attns = torch.cat([atts_bos, speech_atts], dim=1)       # [B, 1+T_new]

        eos_id = self.llama_tokenizer.eos_token_id
        stop_words_ids = [torch.tensor([eos_id], device=self.device)]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            attention_mask=attns,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
        )

        # special tokens(EOS 등)은 빼고 디코드하는 게 일반적
        text = self.llama_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return text


    # ----------------------------------------------------------------------
    # config(dict)에서 바로 초기화할 수 있는 helper
    # ----------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "modelYIM":
        """
        config: 보통 config["model"] 블록을 그대로 넘겨 받는다고 가정.
        예:
            model_cfg = full_cfg["model"]
            model = modelYIM.from_config(model_cfg)
        """
        llama_path = config.get("llama_path", "")
        conformer_path = config.get("conformer_path", "")

        conformer_dim = config.get("conformer_dim", 512)
        conformer_layers = config.get("conformer_layers", 12)
        conformer_input_dim = config.get("conformer_input_dim", 80)
        conformer_concat_num = config.get("conformer_concat_num", 3)

        feed_forward_expansion_factor = config.get("feed_forward_expansion_factor", 4)
        conv_expansion_factor = config.get("conv_expansion_factor", 2)
        conv_kernel_size = config.get("conv_kernel_size", 31)
        dropout = config.get("dropout", 0.1)
        subsampling_factor = config.get("subsampling_factor", 8)
        min_subsample_len_multiplier = config.get("min_subsample_len_multiplier", 2)
        num_attention_heads = config.get("num_attention_heads", 8)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)
        train_llama = config.get("train_llama", False)
        max_txt_len = config.get("max_txt_len", 128)

        model = cls(
            llama_path=llama_path,
            conformer_path=conformer_path,
            conformer_dim=conformer_dim,
            conformer_layers=conformer_layers,
            conformer_input_dim=conformer_input_dim,
            conformer_concat_num=conformer_concat_num,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            subsampling_factor=subsampling_factor,
            min_subsample_len_multiplier=min_subsample_len_multiplier,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            train_llama=train_llama,
            max_txt_len=max_txt_len,
        )


        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load modelYIM ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # stage1 Conformer / LLaMA weight 구조와 충돌하지 않게 strict=False
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)

        return model
            

# class modelYIM(nn.Module):
#     @classmethod
#     def init_conformer(self, input_dim=80, encoder_dim=512, num_encoder_layers=12, modelpath = " "):

#         conformer = Conformer(input_dim=input_dim, 
#                   encoder_dim=encoder_dim, 
#                   num_encoder_layers=num_encoder_layers,
#                   modelpath=modelpath).to(self.device)
#         return conformer
    
#     def maybe_autocast(self, dtype=torch.float16):
#         # if on cpu, don't use autocast
#         # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
#         enable_autocast = self.device != torch.device("cpu")

#         if enable_autocast:
#             return torch.cuda.amp.autocast(dtype=dtype)
#         else:
#             return contextlib.nullcontext()

#     def __init__(
#         self,
#         llama_path="",
#         conformer_path="",
#         conformer_dim=512,
#         conformer_layers=12,
#         conformer_input_dim=80,
#         conformer_concat_num = 3,
#         speech_llama_proj_model="",
#         freeze_speech_llama_proj=False,

#         lora=True,
#         lora_rank=8,
#         lora_alpha=32,
#         lora_dropout=0.1,
#         train_llama: bool = False,

#         max_txt_len=128,
#     ):
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.lora = lora
#         self.train_llama = train_llama
#         self.max_txt_len = max_txt_len

#         logging.info('Loading LLaMA Tokenizer')
#         self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False)
#         self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         self.llama_tokenizer.padding_side = "right"

#         logging.info('Loading LLaMA Model')
#         self.llama_model = LlamaForCausalLM.from_pretrained(
#             llama_path,
#             torch_dtype=torch.float16,
#         )
#         self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
#         logging.info('Loading LLaMA Done')

#         # 🔹 학습 모드 결정 로직
#         if self.lora:
#             # LoRA를 쓰는 경우: base LLaMA는 freeze, LoRA 모듈만 학습
#             for name, param in self.llama_model.named_parameters():
#                 param.requires_grad = False
#             logging.info('Base LLaMA is frozen. LoRA adapters will be trainable.')

#             self.peft_config = LoraConfig(
#                 task_type=TaskType.CAUSAL_LM,
#                 inference_mode=False,
#                 r=lora_rank,
#                 lora_alpha=lora_alpha,
#                 lora_dropout=lora_dropout,
#             )
#             self.llama_model = get_peft_model(self.llama_model, self.peft_config)
#             self.llama_model.print_trainable_parameters()
#             logging.info('LoRA Training Enabled')

#         else:
#             # LoRA를 쓰지 않는 경우: train_llama flag로 전체 LLaMA 학습 여부 결정
#             if not self.train_llama:
#                 for name, param in self.llama_model.named_parameters():
#                     param.requires_grad = False
#                 logging.info('LLaMA is frozen (no LoRA, no full fine-tuning).')
#             else:
#                 logging.info('Full LLaMA fine-tuning is enabled (no LoRA).')


#         assert conformer_path
#         logging.info('Loading Conformer Model')
#         self.conformer = self.init_conformer(modelpath=conformer_path, input_dim=conformer_input_dim, encoder_dim=conformer_dim, num_encoder_layers=conformer_layers)
#         self.ln_speech = nn.LayerNorm(self.conformer.config.d_model)

    
#         logging.info('Loading speech LLAMA proj')
#         logging.info("Initializing speech LLAMA proj")
#         in_dim = conformer_dim * conformer_concat_num
#         out_dim = self.llama_model.config.hidden_size

#         self.speech_llama_proj = SpeechLlamaProj(
#             in_dim=in_dim,
#             out_dim=out_dim,
#             pretrained_path=speech_llama_proj_model,  # 없으면 None
#             freeze=freeze_speech_llama_proj,
#             key_in_ckpt="speech_llama_proj",         # ckpt 구조에 맞게 조정 가능
#         )


#     def forward(self, samples, verbose=False):

#         # use speech/audio encoder to encode speech/audio
#         spectrogram = samples["spectrogram"]
#         raw_wav = samples.get("raw_wav", None)
#         audio_padding_mask = samples.get("padding_mask", None)


#         #(B, T', encoder_dim)
#         # intput shape 만들어야 함 
#         speech_embeds = self.conformer(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

#         # 여기서 위의 speech embeds를 time 마다 conformer_concat_num만큼 concat한 뒤 LLaMA 차원으로 바꿔줘야 함

#         B, T_enc, C_enc = speech_embeds.shape
#         conformer_concat_num = getattr(self, "conformer_concat_num", 1)

#         if conformer_concat_num > 1:
#             # T_enc이 concat_num의 배수가 아닐 수 있으므로 뒤를 잘라서 맞춤
#             T_trim = (T_enc // conformer_concat_num) * conformer_concat_num
#             if T_trim != T_enc:
#                 speech_embeds = speech_embeds[:, :T_trim, :]
#                 T_enc = T_trim

#             T_new = T_enc // conformer_concat_num  # concat 후 토큰 개수
#             # (B, T_enc, C_enc) → (B, T_new, C_enc * concat_num)
#             concat_embeds = speech_embeds.view(
#                 B,
#                 T_new,
#                 C_enc * conformer_concat_num,
#             )
#         else:
#             # concat_num = 1 이면 그대로 사용
#             concat_embeds = speech_embeds
#             T_new = T_enc

#         # 모든 프레임을 유효 토큰으로 사용 → attention mask = 1
#         # (B, T_new)
#         speech_embeds = self.speech_llama_proj(concat_embeds)
#         speech_atts = torch.ones(
#             speech_embeds.size()[:-1],
#             dtype=torch.long,
#             device=self.device,
#         )

#         # prepare inputs for LLM
#         text = [t + self.llama_tokenizer.eos_token for t in samples["text"]]

#         to_regress_tokens = self.llama_tokenizer(
#             text,
#             return_tensors="pt",
#             padding="longest",
#             truncation=True,
#             max_length=self.max_txt_len,
#             add_special_tokens=False
#         ).to(self.device)

#         to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
#         targets = to_regress_tokens.input_ids.masked_fill(
#             to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
#         )
#         # 음성 구간(bos + speech_embeds)에 대해서는 loss를 계산하지 않기 위해
#         # 길이: (B, 1 + T_new), 값은 전부 -100
#         empty_targets = (
#             torch.ones(
#                 [speech_atts.shape[0], speech_atts.shape[1] + 1],
#                 dtype=torch.long,
#                 device=self.device,
#             ).fill_(-100)
#         )
#         # 최종 targets: [bos + speech] 구간은 -100, 그 뒤 텍스트 토큰은 실제 label
#         # shape: (B, 1 + T_new + T_txt)
#         targets = torch.cat([empty_targets, targets], dim=1)


#         # 4) BOS 토큰 + 음성 임베딩 + 텍스트 임베딩 → LLaMA 입력 구성
#         batch_size = speech_embeds.shape[0]

#         # BOS 토큰 id 텐서: (B, 1)
#         bos = torch.ones(
#             [batch_size, 1],
#             dtype=to_regress_tokens.input_ids.dtype,
#             device=self.device,
#         ) * self.llama_tokenizer.bos_token_id

#         bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
#         atts_bos = speech_atts[:, :1]

#         # 최종 입력 임베딩: [BOS] + [speech] + [text]
#         # shape: (B, 1 + T_new + T_txt, hidden_llama)
#         inputs_embeds = torch.cat(
#             [bos_embeds, speech_embeds, to_regress_embeds],
#             dim=1,
#         )

#         # 최종 attention mask: (B, 1 + T_new + T_txt)
#         attention_mask = torch.cat(
#             [atts_bos, speech_atts, to_regress_tokens.attention_mask],
#             dim=1,
#         )

#         # calulate loss
#         with self.maybe_autocast():
#             outputs = self.llama_model(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 return_dict=True,
#                 labels=targets,
#             )
#             loss = outputs.loss

#         if verbose:
#             nvocab = self.llama_model.config.vocab_size
#             results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
#             labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
#             mask = (labels != -100)
#             correct = (results[mask] == labels[mask]).float().sum()
#             total = len(labels[mask])

#         if verbose:
#             return {"loss": loss, "correct": correct, "total": total}

#         return outputs
    
#     def generate(self, samples, generate_cfg, prompts=None):
#         batch_size = samples["spectrogram"].shape[0]

#         spectrogram = samples["spectrogram"]
#         raw_wav = samples.get("raw_wav", None)
#         audio_padding_mask = samples.get("padding_mask", None)


#         #(B, T', encoder_dim)
#         # intput shape 만들어야 함 
#         speech_embeds = self.conformer(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

#         # 여기서 위의 speech embeds를 time 마다 conformer_concat_num만큼 concat한 뒤 LLaMA 차원으로 바꿔줘야 함

#         B, T_enc, C_enc = speech_embeds.shape
#         conformer_concat_num = getattr(self, "conformer_concat_num", 1)

#         if conformer_concat_num > 1:
#             # T_enc이 concat_num의 배수가 아닐 수 있으므로 뒤를 잘라서 맞춤
#             T_trim = (T_enc // conformer_concat_num) * conformer_concat_num
#             if T_trim != T_enc:
#                 speech_embeds = speech_embeds[:, :T_trim, :]
#                 T_enc = T_trim

#             T_new = T_enc // conformer_concat_num  # concat 후 토큰 개수
#             # (B, T_enc, C_enc) → (B, T_new, C_enc * concat_num)
#             concat_embeds = speech_embeds.view(
#                 B,
#                 T_new,
#                 C_enc * conformer_concat_num,
#             )
#         else:
#             # concat_num = 1 이면 그대로 사용
#             concat_embeds = speech_embeds
#             T_new = T_enc

#         # 모든 프레임을 유효 토큰으로 사용 → attention mask = 1
#         # (B, T_new)
#         speech_embeds = self.speech_llama_proj(concat_embeds)
#         speech_atts = torch.ones(
#             speech_embeds.size()[:-1],
#             dtype=torch.long,
#             device=self.device,
#         )
#         bos = torch.ones(
#             [batch_size, 1],
#             dtype=torch.int32,
#             device=speech_embeds.device,
#         ) * self.llama_tokenizer.bos_token_id
#         bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
#         atts_bos = speech_atts[:, :1]

#         embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
#         attns = torch.cat([atts_bos, speech_atts], dim=1)

#         stop_words_ids = [torch.tensor([2]).cuda()]  
#         stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
#         outputs = self.llama_model.generate(
#             inputs_embeds=embeds,
#             max_new_tokens=generate_cfg.get("max_new_tokens", 200),
#             stopping_criteria=stopping_criteria,
#             num_beams=generate_cfg.get("num_beams", 4),
#             do_sample=generate_cfg.get("do_sample", False),
#             min_length=generate_cfg.get("min_length", 1),
#             temperature=generate_cfg.get("temperature", 1.0),
#             top_p=generate_cfg.get("top_p", 0.9),
#             repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
#             length_penalty=generate_cfg.get("length_penalty", 1.0),
#             attention_mask=attns,
#         )
#         text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

#         return text

#     @classmethod
#     def from_config(cls, config):

#         llama_path = config.get("llama_path")
#         conformer_path = config.get("conformer_path", "")

#         conformer_dim = config.get("conformer_dim", 512)
#         conformer_layers = config.get("conformer_layers", 12)
#         conformer_input_dim = config.get("conformer_input_dim", 80)
#         conformer_concat_num = config.get("conformer_concat_num", 3)

#         speech_llama_proj_model = config.get("speech_llama_proj_model", "")
#         freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

#         lora = config.get("lora", True)
#         lora_rank = config.get("lora_rank", 8)
#         lora_alpha = config.get("lora_alpha", 32)
#         lora_dropout = config.get("lora_dropout", 0.1)
#         train_llama = config.get("train_llama", False)   # 🔹 추가
#         max_txt_len = config.get("max_txt_len", 128)

#         model = cls(
#             llama_path=llama_path,
#             conformer_path=conformer_path,
#             conformer_dim=conformer_dim,
#             conformer_layers=conformer_layers,
#             conformer_input_dim=conformer_input_dim,
#             conformer_concat_num=conformer_concat_num,
#             speech_llama_proj_model=speech_llama_proj_model,
#             freeze_speech_llama_proj=freeze_speech_llama_proj,
#             lora=lora,
#             lora_rank=lora_rank,
#             lora_alpha=lora_alpha,
#             lora_dropout=lora_dropout,
#             train_llama=train_llama,
#             max_txt_len=max_txt_len,
#         )

#         ckpt_path = config.get("ckpt", "")
#         if ckpt_path:
#             logging.info("Load modelYIM ckpt from: {}".format(ckpt_path))
#             ckpt = torch.load(ckpt_path, map_location="cpu")
#             model.load_state_dict(ckpt['model'], strict=False)

#         return model
            
