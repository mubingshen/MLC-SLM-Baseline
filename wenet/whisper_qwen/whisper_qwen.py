# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#               2025 ASLP@NPU for MLC-SLM Baseline. (authors: Bingshen Mu)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

from typing import Dict, List, Optional, Tuple
import torch

from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.search import DecodeResult
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from wenet.transformer.subsampling import WhisperProjector

class WhisperQwen(torch.nn.Module):
    """Whisper Encoder + Linear Projector + Qwen LLM Decoder"""

    def __init__(
        self,
        encoder: BaseEncoder,
        qwen: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        use_qwen_lora: bool,
    ):

        super().__init__()
        self.encoder = encoder
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.encoder_projector = WhisperProjector(downsample_rate=4, idim=1280, odim=3584) # whisper output_dim: 1280, qwen2.7-7B hidden size: 3584
        
        self.use_qwen_lora = use_qwen_lora
        self.trainable_prompts = torch.nn.Embedding(20, 3584)

    def prompt_wrap(self, speech_embeds):
        batch_size = speech_embeds.size(0)
        prompt_ids = torch.tensor(list(range(20)), dtype=torch.int64, device=speech_embeds.device)
        prompt_ids = prompt_ids.unsqueeze(0).repeat_interleave(batch_size, dim=0)
        prompt_embeds = self.trainable_prompts(prompt_ids)

        wrapped_embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        return wrapped_embeds
    
    def pad_list(self, xs: List[torch.Tensor], pad_value: int):
        """Perform padding for the list of tensors.

        Args:
            xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
            pad_value (float): Value for padding.

        Returns:
            Tensor: Padded tensor (B, Tmax, `*`).

        Examples:
            >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
            >>> x
            [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
            >>> pad_list(x, 0)
            tensor([[1., 1., 1., 1.],
                    [1., 1., 0., 0.],
                    [1., 0., 0., 0.]])

        """
        max_len = max([len(item) for item in xs])
        batchs = len(xs)
        ndim = xs[0].ndim
        if ndim == 1:
            pad_res = torch.zeros(batchs,
                                max_len,
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        elif ndim == 2:
            pad_res = torch.zeros(batchs,
                                max_len,
                                xs[0].shape[1],
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        elif ndim == 3:
            pad_res = torch.zeros(batchs,
                                max_len,
                                xs[0].shape[1],
                                xs[0].shape[2],
                                dtype=xs[0].dtype,
                                device=xs[0].device)
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")
        pad_res.fill_(pad_value)
        for i in range(batchs):
            pad_res[i, :len(xs[i])] = xs[i]
        return pad_res

    def add_eos(self, ys_pad: torch.Tensor, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    
        _eos = torch.tensor([eos],
                            dtype=torch.long,
                            requires_grad=False,
                            device=ys_pad.device)
        ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [y for y in ys]
        ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
        return self.pad_list(ys_in, eos), self.pad_list(ys_out, ignore_id)

    @torch.jit.unused
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        
        # 1. Encoder
        encoder_out, _ = self.encoder(speech, speech_lengths) # 2 times subsampling
        
        
        # 2. WhisperProjector
        speech_embeds = self.encoder_projector(encoder_out) # 4 times subsampling, 8 times subsampling totally
        
        # 3. wrap speech_embeds with prompts
        speech_embeds = self.prompt_wrap(speech_embeds)
        
        # 4. prepare inputs for qwen
        to_regress_tokens_in, to_regress_tokens_out = self.add_eos(text, self.qwen.config.eos_token_id, -100)
        to_regress_embeds = self.qwen.model.embed_tokens(to_regress_tokens_in.to(speech.device)) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(to_regress_tokens_in.to(speech.device))

        bos = torch.ones(
            [speech_embeds.shape[0], 1],
            dtype=torch.int64,
            device=to_regress_embeds.device,
        ) * self.qwen.config.bos_token_id # bos_token_id: 151643
        bos_embeds = self.qwen.model.embed_tokens(bos) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(bos)

        eos = torch.ones(
            [speech_embeds.shape[0], 1],
            dtype=torch.int64,
            device=to_regress_embeds.device,
        ) * self.qwen.config.eos_token_id # eos_token_id: 151643
        eos_embeds = self.qwen.model.embed_tokens(eos) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(eos)

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds, eos_embeds], dim=1).to(torch.bfloat16)
        
        empty_targets = (
            torch.ones(
                [speech_embeds.shape[0], speech_embeds.shape[1] + 1],
                dtype=torch.long
            ).to(speech.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, to_regress_tokens_out.to(speech.device)], dim=1)
        
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}
    
    def decode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        qwen_path: str,
    ) -> Dict[str, List[DecodeResult]]:
        
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths) # 2 times subsampling
        
        # 2. WhisperProjector
        speech_embeds = self.encoder_projector(encoder_out) # 4 times subsampling, 8 times subsampling totally
        
        # 3. wrap speech_embeds with prompts
        speech_embeds = self.prompt_wrap(speech_embeds)
        
        # 4. prepare inputs for qwen
        bos = torch.ones(
            [speech_embeds.shape[0], 1],
            dtype=torch.int64,
            device=speech_embeds.device,
        ) * self.qwen.config.bos_token_id # bos_token_id: 128000
        bos_embeds = self.qwen.model.embed_tokens(bos) if not self.use_qwen_lora else self.qwen.model.model.embed_tokens(bos)

        inputs_embeds = torch.cat([bos_embeds, speech_embeds], dim=1).to(torch.bfloat16)
        
        # TODO: different decoding params
        self.qwen.generation_config = GenerationConfig.from_pretrained(qwen_path, do_sample=False, max_new_tokens=200, num_beams=1, min_length=1, temperature=1.0, repetition_penalty=1.0, length_penalty=1.0)
        # import pdb;pdb.set_trace()
        results = self.qwen.generate(
            inputs_embeds=inputs_embeds,
            generation_config=self.qwen.generation_config
        )
        results = self.tokenizer.batch_decode(results, skip_special_tokens=True)
        
        return results
