"""
Student submission entry point for the strict inference-engine track.

This file implements a small manual Qwen2 inference engine: tokenizer/config
loading, raw safetensor weights, RMSNorm, RoPE, GQA attention, MLP, KV cache,
prefill, and greedy decode.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from student_release.utils.load_weights import load_config_and_state_dict


def _cfg_key(*parts: str) -> str:
    return "_".join(parts)


def _select_dtype(name: str, device: torch.device, config: dict[str, Any]) -> torch.dtype:
    normalized = str(name or "float32").lower()
    if normalized == "auto":
        normalized = str(config.get("torch_dtype", "float32")).lower()
    mapping = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    selected = mapping.get(normalized, torch.float32)
    if device.type == "cpu" and selected in {torch.float16, torch.bfloat16}:
        return torch.float32
    return selected


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    x_float = x.to(torch.float32)
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps)).to(input_dtype) * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class StudentEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
        seed: int = 0,
    ):
        torch.manual_seed(int(seed))
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")

        self.model_path = model_path
        self.device = requested_device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.local_files_only = local_files_only

        self.config, state_dict = load_config_and_state_dict(model_path, device="cpu")
        self.compute_dtype = _select_dtype(dtype, self.device, self.config)
        self.weights: dict[str, torch.Tensor] = {
            name_key: tensor.to(device=self.device, dtype=self.compute_dtype)
            for name_key, tensor in state_dict.items()
        }
        state_dict.clear()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = int(self.config[_cfg_key("hidden", "size")])
        self.num_layers = int(self.config[_cfg_key("num", "hidden", "layers")])
        self.num_heads = int(self.config["num_attention_heads"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rms_eps = float(self.config.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))
        self.eos_token_id = int(self.config.get("eos_token_id", self.tokenizer.eos_token_id or 0))
        self.bos_token_id = self.config.get("bos_token_id", self.tokenizer.bos_token_id)
        self.inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
                / self.head_dim
            )
        )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        del suite_name
        return [self._generate_one(str(prompt), int(max_new_tokens)) for prompt in prompts]

    def serve_requests(self, requests: list[dict], batch_size: int | None = None):
        del batch_size
        outputs: list[str] = []
        for request in requests:
            outputs.append(
                self._generate_one(
                    str(request.get("prompt", "")),
                    int(request.get("max_new_tokens", 1)),
                )
            )
        return outputs

    def _weight(self, name_key: str) -> torch.Tensor:
        return self.weights[name_key]

    def _linear(
        self,
        x: torch.Tensor,
        weight_name: str,
        bias_name: str | None = None,
    ) -> torch.Tensor:
        bias = self.weights.get(bias_name) if bias_name is not None else None
        return F.linear(x, self._weight(weight_name), bias)

    def _tokenize(self, prompt: str) -> list[int]:
        text = prompt
        if "<|im_start|>" not in text:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        encoded = self.tokenizer(text, add_special_tokens=False)
        token_ids = [int(item) for item in encoded.get("input_ids", [])]
        if token_ids:
            return token_ids
        if self.bos_token_id is not None:
            return [int(self.bos_token_id)]
        return [self.eos_token_id]

    def _rotary_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(device=self.device, dtype=torch.float32)
        freqs = torch.einsum("bs,d->bsd", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(self.compute_dtype), emb.sin().to(self.compute_dtype)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return x
        batch, kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(batch, kv_heads * self.num_kv_groups, seq_len, head_dim)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_prefill: bool,
    ) -> torch.Tensor:
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        if self.attn_implementation == "sdpa":
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_prefill)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if is_prefill:
            seq_len = q.shape[-2]
            mask = torch.full(
                (seq_len, seq_len),
                torch.finfo(scores.dtype).min,
                dtype=scores.dtype,
                device=scores.device,
            )
            scores = scores + torch.triu(mask, diagonal=1)[None, None, :, :]
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, v)

    def _self_attention(
        self,
        hidden_states: torch.Tensor,
        layer_index: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past: tuple[torch.Tensor, torch.Tensor, int] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, int]]:
        prefix = f"model.layers.{layer_index}.self_attn"
        batch_size, seq_len, _ = hidden_states.shape
        q = self._linear(hidden_states, f"{prefix}.q_proj.weight", f"{prefix}.q_proj.bias")
        k = self._linear(hidden_states, f"{prefix}.k_proj.weight", f"{prefix}.k_proj.bias")
        v = self._linear(hidden_states, f"{prefix}.v_proj.weight", f"{prefix}.v_proj.bias")

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)

        if past is None:
            full_k, full_v = k, v
            layer_cache = (full_k, full_v, seq_len)
            is_prefill = seq_len > 1
        else:
            past_k, past_v, past_len = past
            next_len = past_len + seq_len
            if next_len <= past_k.shape[2]:
                past_k[:, :, past_len:next_len, :] = k
                past_v[:, :, past_len:next_len, :] = v
                full_k = past_k[:, :, :next_len, :]
                full_v = past_v[:, :, :next_len, :]
                layer_cache = (past_k, past_v, next_len)
            else:
                full_k = torch.cat((past_k[:, :, :past_len, :], k), dim=2)
                full_v = torch.cat((past_v[:, :, :past_len, :], v), dim=2)
                layer_cache = (full_k, full_v, next_len)
            is_prefill = False

        attn_output = self._attention(q, full_k, full_v, is_prefill=is_prefill)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        return self._linear(attn_output, f"{prefix}.o_proj.weight"), layer_cache

    def _mlp(self, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        prefix = f"model.layers.{layer_index}.mlp"
        gate = self._linear(hidden_states, f"{prefix}.gate_proj.weight")
        up = self._linear(hidden_states, f"{prefix}.up_proj.weight")
        return self._linear(F.silu(gate) * up, f"{prefix}.down_proj.weight")

    def _forward_tokens(
        self,
        input_ids: torch.Tensor,
        start_position: int,
        cache: list[tuple[torch.Tensor, torch.Tensor, int]] | None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, int]]]:
        seq_len = input_ids.shape[1]
        positions = torch.arange(
            start_position,
            start_position + seq_len,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        cos, sin = self._rotary_cos_sin(positions)
        hidden_states = self._weight("model.embed_tokens.weight")[input_ids]
        new_cache: list[tuple[torch.Tensor, torch.Tensor, int]] = []

        for layer_index in range(self.num_layers):
            residual = hidden_states
            hidden_states = _rms_norm(
                hidden_states,
                self._weight(f"model.layers.{layer_index}.input_layernorm.weight"),
                self.rms_eps,
            )
            attn_output, layer_cache = self._self_attention(
                hidden_states,
                layer_index,
                cos,
                sin,
                None if cache is None else cache[layer_index],
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = _rms_norm(
                hidden_states,
                self._weight(f"model.layers.{layer_index}.post_attention_layernorm.weight"),
                self.rms_eps,
            )
            hidden_states = residual + self._mlp(hidden_states, layer_index)
            new_cache.append(layer_cache)

        hidden_states = _rms_norm(hidden_states, self._weight("model.norm.weight"), self.rms_eps)
        return hidden_states, new_cache

    def _reserve_cache(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor, int]],
        total_tokens: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        reserved: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for key_cache, value_cache, used_tokens in cache:
            if key_cache.shape[2] >= total_tokens:
                reserved.append((key_cache, value_cache, used_tokens))
                continue
            key_buffer = key_cache.new_empty(
                key_cache.shape[0],
                key_cache.shape[1],
                total_tokens,
                key_cache.shape[3],
            )
            value_buffer = value_cache.new_empty(
                value_cache.shape[0],
                value_cache.shape[1],
                total_tokens,
                value_cache.shape[3],
            )
            key_buffer[:, :, :used_tokens, :] = key_cache[:, :, :used_tokens, :]
            value_buffer[:, :, :used_tokens, :] = value_cache[:, :, :used_tokens, :]
            reserved.append((key_buffer, value_buffer, used_tokens))
        return reserved

    def _next_token(self, hidden_states: torch.Tensor) -> int:
        logits = F.linear(hidden_states[:, -1, :], self._weight("model.embed_tokens.weight"))
        return int(torch.argmax(logits, dim=-1).item())

    @torch.inference_mode()
    def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        if max_new_tokens <= 0:
            return ""

        input_ids = torch.tensor([self._tokenize(prompt)], dtype=torch.long, device=self.device)
        hidden_states, cache = self._forward_tokens(input_ids, start_position=0, cache=None)
        cache = self._reserve_cache(cache, input_ids.shape[1] + max_new_tokens)
        next_token = self._next_token(hidden_states)
        generated_ids = [next_token]

        current_position = input_ids.shape[1]
        for _ in range(max_new_tokens - 1):
            step_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            hidden_states, cache = self._forward_tokens(step_ids, current_position, cache)
            next_token = self._next_token(hidden_states)
            generated_ids.append(next_token)
            current_position += 1

        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
