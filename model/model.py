#jhem


import logging
import json
import contextlib
import random

from deep.ref.SALMONN.models.utils import StoppingCriteriaSub
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizer, StoppingCriteriaList, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from conformer import Conformer
from .SpeechLlamaProj import SpeechLlamaProj

class modelYIM(nn.Module):
    @classmethod
    def init_conformer(self, input_dim=80, encoder_dim=512, num_encoder_layers=12, modelpath = " "):

        conformer = Conformer(input_dim=input_dim, 
                  encoder_dim=encoder_dim, 
                  num_encoder_layers=num_encoder_layers,
                  modelpath=modelpath).to(self.device)
        return conformer
    
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        conformer_path="",
        conformer_dim=512,
        conformer_layers=12,
        conformer_input_dim=80,
        conformer_concat_num = 3,
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        max_txt_len=128,
        end_sym="</s>",
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora = lora
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llama_tokenizer.padding_side = "right"

        logging.info('Loading LLaMA Model')
        self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.float16,
        )

        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLaMA Done')

        if self.lora:
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=lora_rank, 
                lora_alpha=lora_alpha, 
                lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            self.llama_model.print_trainable_parameters()
            logging.info('LoRA Training')

        assert conformer_path
        logging.info('Loading Conformer Model')
        self.conformer = self.init_conformer(modelpath=conformer_path, input_dim=conformer_input_dim, encoder_dim=conformer_dim, num_encoder_layers=conformer_layers)
        self.ln_speech = nn.LayerNorm(self.conformer.config.d_model)

    
        logging.info('Loading speech LLAMA proj')
        logging.info("Initializing speech LLAMA proj")
        in_dim = conformer_dim * conformer_concat_num
        out_dim = self.llama_model.config.hidden_size

        self.speech_llama_proj = SpeechLlamaProj(
            in_dim=in_dim,
            out_dim=out_dim,
            pretrained_path=speech_llama_proj_model,  # 없으면 None
            freeze=freeze_speech_llama_proj,
            key_in_ckpt="speech_llama_proj",         # ckpt 구조에 맞게 조정 가능
        )


    def forward(self, samples, verbose=False):

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)


        #(B, T', encoder_dim)
        # intput shape 만들어야 함 
        speech_embeds = self.conformer(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # 여기서 위의 speech embeds를 time 마다 conformer_concat_num만큼 concat한 뒤 LLaMA 차원으로 바꿔줘야 함

        B, T_enc, C_enc = speech_embeds.shape
        conformer_concat_num = getattr(self, "conformer_concat_num", 1)

        if conformer_concat_num > 1:
            # T_enc이 concat_num의 배수가 아닐 수 있으므로 뒤를 잘라서 맞춤
            T_trim = (T_enc // conformer_concat_num) * conformer_concat_num
            if T_trim != T_enc:
                speech_embeds = speech_embeds[:, :T_trim, :]
                T_enc = T_trim

            T_new = T_enc // conformer_concat_num  # concat 후 토큰 개수
            # (B, T_enc, C_enc) → (B, T_new, C_enc * concat_num)
            concat_embeds = speech_embeds.view(
                B,
                T_new,
                C_enc * conformer_concat_num,
            )
        else:
            # concat_num = 1 이면 그대로 사용
            concat_embeds = speech_embeds
            T_new = T_enc

        # 모든 프레임을 유효 토큰으로 사용 → attention mask = 1
        # (B, T_new)
        speech_embeds = self.speech_llama_proj(concat_embeds)
        speech_atts = torch.ones(
            speech_embeds.size()[:-1],
            dtype=torch.long,
            device=self.device,
        )

        # prepare inputs for LLM
        text = [t + self.llama_tokenizer.eos_token for t in samples["text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(self.device)

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        # 음성 구간(bos + speech_embeds)에 대해서는 loss를 계산하지 않기 위해
        # 길이: (B, 1 + T_new), 값은 전부 -100
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long,
                device=self.device,
            ).fill_(-100)
        )
        # 최종 targets: [bos + speech] 구간은 -100, 그 뒤 텍스트 토큰은 실제 label
        # shape: (B, 1 + T_new + T_txt)
        targets = torch.cat([empty_targets, targets], dim=1)


        # --------------------------------------------------
        # 4) BOS 토큰 + 음성 임베딩 + 텍스트 임베딩 → LLaMA 입력 구성
        # --------------------------------------------------
        batch_size = speech_embeds.shape[0]

        # BOS 토큰 id 텐서: (B, 1)
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=self.device,
        ) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        # 최종 입력 임베딩: [BOS] + [speech] + [text]
        # shape: (B, 1 + T_new + T_txt, hidden_llama)
        inputs_embeds = torch.cat(
            [bos_embeds, speech_embeds, to_regress_embeds],
            dim=1,
        )

        # 최종 attention mask: (B, 1 + T_new + T_txt)
        attention_mask = torch.cat(
            [atts_bos, speech_atts, to_regress_tokens.attention_mask],
            dim=1,
        )

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return outputs
    
    def generate(self, samples, generate_cfg, prompts=None):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)


        #(B, T', encoder_dim)
        # intput shape 만들어야 함 
        speech_embeds = self.conformer(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # 여기서 위의 speech embeds를 time 마다 conformer_concat_num만큼 concat한 뒤 LLaMA 차원으로 바꿔줘야 함

        B, T_enc, C_enc = speech_embeds.shape
        conformer_concat_num = getattr(self, "conformer_concat_num", 1)

        if conformer_concat_num > 1:
            # T_enc이 concat_num의 배수가 아닐 수 있으므로 뒤를 잘라서 맞춤
            T_trim = (T_enc // conformer_concat_num) * conformer_concat_num
            if T_trim != T_enc:
                speech_embeds = speech_embeds[:, :T_trim, :]
                T_enc = T_trim

            T_new = T_enc // conformer_concat_num  # concat 후 토큰 개수
            # (B, T_enc, C_enc) → (B, T_new, C_enc * concat_num)
            concat_embeds = speech_embeds.view(
                B,
                T_new,
                C_enc * conformer_concat_num,
            )
        else:
            # concat_num = 1 이면 그대로 사용
            concat_embeds = speech_embeds
            T_new = T_enc

        # 모든 프레임을 유효 토큰으로 사용 → attention mask = 1
        # (B, T_new)
        speech_embeds = self.speech_llama_proj(concat_embeds)
        speech_atts = torch.ones(
            speech_embeds.size()[:-1],
            dtype=torch.long,
            device=self.device,
        )
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).cuda()]  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text

    @classmethod
    def from_config(cls, config):

        llama_path = config.get("llama_path")
        conformer_path = config.get("conformer_path", "")

        conformer_dim = config.get("conformer_dim", 512)
        conformer_layers = config.get("conformer_layers", 12)
        conformer_input_dim = config.get("conformer_input_dim", 80)
        conformer_concat_num = config.get("conformer_concat_num", 3)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        max_txt_len = config.get("max_txt_len", 128)

        model = cls(
            llama_path=llama_path,
            conformer_path=conformer_path,
            conformer_dim=conformer_dim,
            conformer_layers=conformer_layers,
            conformer_input_dim=conformer_input_dim,
            conformer_concat_num=conformer_concat_num,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            max_txt_len=max_txt_len,
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt['model'], strict=False)

        return model
            
