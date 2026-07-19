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
        self.batch_length_ratio = 1.6
        self.inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
                / self.head_dim
            )
        )
        self.template_prefix_token_ids = self._build_template_prefix_token_ids()
        self.batch_prefix_min_tokens = max(128, len(self.template_prefix_token_ids) + 32)
        self.max_cached_prefix_tokens = min(int(self.config.get("max_position_embeddings", 4096)), 4096)
        self.max_prefix_cache_entries = 4
        self.prefix_cache: dict[
            tuple[int, ...],
            tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, int]]],
        ] = {}
        self.prefix_cache_order: list[tuple[int, ...]] = []
        self.pinned_prefix_keys: set[tuple[int, ...]] = set()

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        del suite_name
        if not prompts:
            return []
        return self._generate_length_aware(
            [str(prompt) for prompt in prompts],
            int(max_new_tokens),
            max(int(batch_size or 1), 1),
        )

    def serve_requests(self, requests: list[dict], batch_size: int | None = None):
        if not requests:
            return []

        outputs = [""] * len(requests)
        pending: list[dict[str, Any]] = []
        for index, request in enumerate(requests):
            max_new_tokens = max(int(request.get("max_new_tokens", 1)), 0)
            if max_new_tokens <= 0:
                continue
            prompt = str(request.get("prompt", ""))
            pending.append(
                {
                    "index": index,
                    "arrival": float(request.get("arrival_time_ms", index * 10.0)),
                    "priority": int(request.get("priority", 0)),
                    "group_id": str(request.get("group_id", "")),
                    "token_ids": self._tokenize(prompt),
                    "max_new_tokens": max_new_tokens,
                    "generated_ids": [],
                    "generated_count": 0,
                }
            )

        if not pending:
            return outputs

        max_active = self._serving_max_active(requests, batch_size)
        pending.sort(key=lambda item: (item["arrival"], -item["priority"], item["index"]))
        virtual_time_ms = float(pending[0]["arrival"])
        active: list[dict[str, Any]] = []
        active_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        active_lengths: torch.Tensor | None = None
        active_next_tokens: torch.Tensor | None = None

        while pending or active:
            if not active and pending and float(pending[0]["arrival"]) > virtual_time_ms:
                virtual_time_ms = float(pending[0]["arrival"])

            slots = max_active - len(active)
            if slots > 0:
                arriving, pending = self._pop_arrived_requests(pending, virtual_time_ms, slots)
                if arriving:
                    (
                        admitted,
                        new_cache,
                        new_lengths,
                        new_next_tokens,
                    ) = self._prefill_serving_requests(arriving, outputs)
                    if admitted:
                        active_cache = self._append_active_cache_rows(active_cache, new_cache)
                        active_lengths = (
                            new_lengths
                            if active_lengths is None
                            else torch.cat((active_lengths, new_lengths), dim=0)
                        )
                        active_next_tokens = (
                            new_next_tokens
                            if active_next_tokens is None
                            else torch.cat((active_next_tokens, new_next_tokens), dim=0)
                        )
                        active.extend(admitted)

            if not active:
                continue

            assert active_cache is not None
            assert active_lengths is not None
            assert active_next_tokens is not None
            hidden_states, active_cache, active_lengths = self._decode_step_batch(
                active_next_tokens,
                active_cache,
                active_lengths,
            )
            candidate_next_tokens = self._next_tokens(hidden_states)

            keep_positions: list[int] = []
            next_token_values: list[int] = []
            for active_position, item in enumerate(active):
                token_id = int(candidate_next_tokens[active_position].item())
                item["generated_ids"].append(token_id)
                item["generated_count"] += 1
                if int(item["generated_count"]) >= int(item["max_new_tokens"]):
                    outputs[int(item["index"])] = self._decode_generated_ids(item["generated_ids"])
                else:
                    keep_positions.append(active_position)
                    next_token_values.append(token_id)

            if keep_positions:
                active = [active[position] for position in keep_positions]
                active_cache = self._select_active_cache_rows(active_cache, keep_positions)
                keep_tensor = torch.tensor(keep_positions, dtype=torch.long, device=self.device)
                active_lengths = active_lengths.index_select(0, keep_tensor)
                active_next_tokens = torch.tensor(
                    next_token_values,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                active = []
                active_cache = None
                active_lengths = None
                active_next_tokens = None

            virtual_time_ms += 1.0
        return outputs

    def _serving_max_active(self, requests: list[dict], batch_size: int | None) -> int:
        if batch_size is not None:
            return max(int(batch_size), 1)
        stream_size = 0
        for request in requests:
            stream_size = max(stream_size, int(request.get("stream_size", 0) or 0))
        return max(stream_size or len(requests), 1)

    def _pop_arrived_requests(
        self,
        pending: list[dict[str, Any]],
        virtual_time_ms: float,
        slots: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        arrived = [item for item in pending if float(item["arrival"]) <= virtual_time_ms]
        if not arrived:
            return [], pending
        arrived.sort(
            key=lambda item: (
                -int(item["priority"]),
                str(item["group_id"]),
                float(item["arrival"]),
                int(item["index"]),
            )
        )
        selected = arrived[:slots]
        selected_indices = {int(item["index"]) for item in selected}
        remaining = [item for item in pending if int(item["index"]) not in selected_indices]
        remaining.sort(key=lambda item: (item["arrival"], -item["priority"], item["index"]))
        return selected, remaining

    def _prefill_serving_requests(
        self,
        items: list[dict[str, Any]],
        outputs: list[str],
    ) -> tuple[
        list[dict[str, Any]],
        list[list[tuple[torch.Tensor, torch.Tensor, int]]],
        torch.Tensor,
        torch.Tensor,
    ]:
        self._prepare_batch_prefix_cache([item["token_ids"] for item in items])
        admitted: list[dict[str, Any]] = []
        new_caches: list[list[tuple[torch.Tensor, torch.Tensor, int]]] = []
        lengths: list[int] = []
        next_tokens: list[int] = []
        for item in items:
            token_ids = item["token_ids"]
            max_new_tokens = int(item["max_new_tokens"])
            hidden_states, cache = self._prefill_single(token_ids, len(token_ids) + max_new_tokens)
            next_token = self._next_token(hidden_states)
            item["generated_ids"] = [next_token]
            item["generated_count"] = 1
            if max_new_tokens <= 1:
                outputs[int(item["index"])] = self._decode_generated_ids(item["generated_ids"])
                continue
            admitted.append(item)
            new_caches.append(cache)
            lengths.append(len(token_ids))
            next_tokens.append(next_token)

        return (
            admitted,
            new_caches,
            torch.tensor(lengths, dtype=torch.long, device=self.device),
            torch.tensor(next_tokens, dtype=torch.long, device=self.device),
        )

    def _decode_generated_ids(self, generated_ids: list[int]) -> str:
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

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

    def _build_template_prefix_token_ids(self) -> tuple[int, ...]:
        samples = ["A", "z", "7"]
        tokenized_samples: list[list[int]] = []
        for sample in samples:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": sample}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = self.tokenizer(text, add_special_tokens=False)
            tokenized_samples.append([int(item) for item in encoded.get("input_ids", [])])
        prefix_len = self._common_prefix_length(tokenized_samples)
        return tuple(tokenized_samples[0][:prefix_len])

    def _common_prefix_length(self, tokenized: list[list[int]]) -> int:
        if not tokenized:
            return 0
        shortest = min(len(token_ids) for token_ids in tokenized)
        prefix_len = 0
        for index in range(shortest):
            token_id = tokenized[0][index]
            if any(other[index] != token_id for other in tokenized[1:]):
                break
            prefix_len += 1
        return prefix_len

    def _is_prefix(self, prefix_key: tuple[int, ...], token_ids: list[int] | tuple[int, ...]) -> bool:
        return len(prefix_key) <= len(token_ids) and tuple(token_ids[: len(prefix_key)]) == prefix_key

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
        query_start: int = 0,
    ) -> torch.Tensor:
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        if self.attn_implementation == "sdpa":
            if not is_prefill:
                return F.scaled_dot_product_attention(q, k, v, is_causal=False)
            if query_start == 0 and q.shape[-2] == k.shape[-2]:
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_mask = self._causal_suffix_mask(
                q.shape[-2],
                k.shape[-2],
                int(query_start),
                q.dtype,
                q.device,
            )
            return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)

        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if is_prefill:
            mask = self._causal_suffix_mask(
                q.shape[-2],
                k.shape[-2],
                int(query_start),
                scores.dtype,
                scores.device,
            )
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return torch.matmul(probs, v)

    def _causal_suffix_mask(
        self,
        query_len: int,
        key_len: int,
        query_start: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(
            query_start,
            query_start + query_len,
            dtype=torch.long,
            device=device,
        ).view(query_len, 1)
        key_positions = torch.arange(key_len, dtype=torch.long, device=device).view(1, key_len)
        invalid = key_positions > query_positions
        mask = torch.zeros((query_len, key_len), dtype=dtype, device=device)
        mask = mask.masked_fill(invalid, torch.finfo(dtype).min)
        return mask.view(1, 1, query_len, key_len)

    def _batch_decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        positions = torch.arange(k.shape[2], device=self.device).view(1, 1, 1, -1)
        invalid = positions >= valid_lengths.view(-1, 1, 1, 1)
        scores = scores.masked_fill(invalid, torch.finfo(scores.dtype).min)
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
            query_start = 0
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
            is_prefill = seq_len > 1
            query_start = past_len

        attn_output = self._attention(q, full_k, full_v, is_prefill=is_prefill, query_start=query_start)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        return self._linear(attn_output, f"{prefix}.o_proj.weight"), layer_cache

    def _mlp(self, hidden_states: torch.Tensor, layer_index: int) -> torch.Tensor:
        prefix = f"model.layers.{layer_index}.mlp"
        gate = self._linear(hidden_states, f"{prefix}.gate_proj.weight")
        up = self._linear(hidden_states, f"{prefix}.up_proj.weight")
        return self._linear(F.silu(gate) * up, f"{prefix}.down_proj.weight")

    def _self_attention_batch_decode(
        self,
        hidden_states: torch.Tensor,
        layer_index: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_cache: tuple[torch.Tensor, torch.Tensor],
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        prefix = f"model.layers.{layer_index}.self_attn"
        batch_size, seq_len, _ = hidden_states.shape
        q = self._linear(hidden_states, f"{prefix}.q_proj.weight", f"{prefix}.q_proj.bias")
        k = self._linear(hidden_states, f"{prefix}.k_proj.weight", f"{prefix}.k_proj.bias")
        v = self._linear(hidden_states, f"{prefix}.v_proj.weight", f"{prefix}.v_proj.bias")

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)

        key_cache, value_cache = layer_cache
        batch_indices = torch.arange(batch_size, device=self.device)
        key_cache[batch_indices, :, lengths, :] = k[:, :, 0, :]
        value_cache[batch_indices, :, lengths, :] = v[:, :, 0, :]

        valid_lengths = lengths + 1
        max_valid_length = int(valid_lengths.max().item())
        full_k = key_cache[:, :, :max_valid_length, :]
        full_v = value_cache[:, :, :max_valid_length, :]

        attn_output = self._batch_decode_attention(q, full_k, full_v, valid_lengths)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        return self._linear(attn_output, f"{prefix}.o_proj.weight")

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

    def _copy_cache_with_capacity(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor, int]],
        total_tokens: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        copied: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for key_cache, value_cache, used_tokens in cache:
            capacity = max(int(total_tokens), int(used_tokens))
            key_buffer = key_cache.new_empty(
                key_cache.shape[0],
                key_cache.shape[1],
                capacity,
                key_cache.shape[3],
            )
            value_buffer = value_cache.new_empty(
                value_cache.shape[0],
                value_cache.shape[1],
                capacity,
                value_cache.shape[3],
            )
            key_buffer[:, :, :used_tokens, :] = key_cache[:, :, :used_tokens, :]
            value_buffer[:, :, :used_tokens, :] = value_cache[:, :, :used_tokens, :]
            copied.append((key_buffer, value_buffer, used_tokens))
        return copied

    def _touch_prefix_cache(self, prefix_key: tuple[int, ...]) -> None:
        self.prefix_cache_order = [key for key in self.prefix_cache_order if key != prefix_key]
        self.prefix_cache_order.append(prefix_key)

    def _evict_prefix_cache(self) -> None:
        while len(self.prefix_cache) > self.max_prefix_cache_entries:
            victim = next(
                (key for key in self.prefix_cache_order if key not in self.pinned_prefix_keys),
                None,
            )
            if victim is None:
                return
            self.prefix_cache.pop(victim, None)
            self.prefix_cache_order = [key for key in self.prefix_cache_order if key != victim]

    def _get_or_build_prefix_cache(
        self,
        prefix_key: tuple[int, ...],
        pinned: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, int]]] | None:
        if not prefix_key:
            return None
        if len(prefix_key) > self.max_cached_prefix_tokens:
            prefix_key = prefix_key[: self.max_cached_prefix_tokens]

        existing = self.prefix_cache.get(prefix_key)
        if existing is not None:
            if pinned:
                self.pinned_prefix_keys.add(prefix_key)
            self._touch_prefix_cache(prefix_key)
            return existing

        input_ids = torch.tensor([list(prefix_key)], dtype=torch.long, device=self.device)
        hidden_states, cache = self._forward_tokens(input_ids, start_position=0, cache=None)
        used_tokens = len(prefix_key)
        compact_cache = [
            (
                key_cache[:, :, :used_tokens, :].contiguous(),
                value_cache[:, :, :used_tokens, :].contiguous(),
                used_tokens,
            )
            for key_cache, value_cache, _ in cache
        ]
        entry = (hidden_states[:, -1:, :].contiguous(), compact_cache)
        self.prefix_cache[prefix_key] = entry
        if pinned:
            self.pinned_prefix_keys.add(prefix_key)
        self._touch_prefix_cache(prefix_key)
        self._evict_prefix_cache()
        return entry

    def _ensure_template_prefix_cache(self) -> None:
        if self.template_prefix_token_ids:
            self._get_or_build_prefix_cache(self.template_prefix_token_ids, pinned=True)

    def _best_cached_prefix_key(self, token_ids: list[int]) -> tuple[int, ...] | None:
        best_key: tuple[int, ...] | None = None
        for prefix_key in self.prefix_cache:
            if best_key is not None and len(prefix_key) <= len(best_key):
                continue
            if self._is_prefix(prefix_key, token_ids):
                best_key = prefix_key
        if best_key is not None:
            self._touch_prefix_cache(best_key)
        return best_key

    def _prepare_batch_prefix_cache(self, tokenized: list[list[int]]) -> None:
        self._ensure_template_prefix_cache()
        if len(tokenized) < 2:
            return
        prefix_len = min(
            self._common_prefix_length(tokenized),
            self.max_cached_prefix_tokens,
        )
        if prefix_len >= self.batch_prefix_min_tokens:
            self._get_or_build_prefix_cache(tuple(tokenized[0][:prefix_len]))

    def _prefill_single(
        self,
        token_ids: list[int],
        total_capacity: int,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, int]]]:
        total_capacity = max(int(total_capacity), len(token_ids))
        self._ensure_template_prefix_cache()
        prefix_key = self._best_cached_prefix_key(token_ids)
        if prefix_key is not None:
            cached_hidden, cached_cache = self.prefix_cache[prefix_key]
            prefix_len = len(prefix_key)
            cache = self._copy_cache_with_capacity(cached_cache, total_capacity)
            suffix_ids = token_ids[prefix_len:]
            if not suffix_ids:
                return cached_hidden, cache
            input_ids = torch.tensor([suffix_ids], dtype=torch.long, device=self.device)
            hidden_states, cache = self._forward_tokens(
                input_ids,
                start_position=prefix_len,
                cache=cache,
            )
            return hidden_states, cache

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        hidden_states, cache = self._forward_tokens(input_ids, start_position=0, cache=None)
        return hidden_states, self._reserve_cache(cache, total_capacity)

    def _merge_caches(
        self,
        single_caches: list[list[tuple[torch.Tensor, torch.Tensor, int]]],
        total_capacity: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        batch_size = len(single_caches)
        merged: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index in range(self.num_layers):
            source_key, source_value, _ = single_caches[0][layer_index]
            key_cache = source_key.new_empty(
                batch_size,
                source_key.shape[1],
                total_capacity,
                source_key.shape[3],
            )
            value_cache = source_value.new_empty(
                batch_size,
                source_value.shape[1],
                total_capacity,
                source_value.shape[3],
            )
            for batch_index, cache in enumerate(single_caches):
                key_source, value_source, used_tokens = cache[layer_index]
                key_cache[batch_index, :, :used_tokens, :] = key_source[0, :, :used_tokens, :]
                value_cache[batch_index, :, :used_tokens, :] = value_source[0, :, :used_tokens, :]
            merged.append((key_cache, value_cache))
        return merged

    def _append_active_cache_rows(
        self,
        active_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
        new_caches: list[list[tuple[torch.Tensor, torch.Tensor, int]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if active_cache is None:
            target_capacity = max(cache[0][0].shape[2] for cache in new_caches)
            return self._merge_caches(new_caches, target_capacity)

        old_batch = active_cache[0][0].shape[0]
        new_batch = len(new_caches)
        target_capacity = max(
            active_cache[0][0].shape[2],
            max(cache[0][0].shape[2] for cache in new_caches),
        )
        merged: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index in range(self.num_layers):
            old_key, old_value = active_cache[layer_index]
            key_cache = old_key.new_empty(
                old_batch + new_batch,
                old_key.shape[1],
                target_capacity,
                old_key.shape[3],
            )
            value_cache = old_value.new_empty(
                old_batch + new_batch,
                old_value.shape[1],
                target_capacity,
                old_value.shape[3],
            )
            key_cache[:old_batch, :, : old_key.shape[2], :] = old_key
            value_cache[:old_batch, :, : old_value.shape[2], :] = old_value
            for offset, cache in enumerate(new_caches):
                key_source, value_source, used_tokens = cache[layer_index]
                row = old_batch + offset
                key_cache[row, :, :used_tokens, :] = key_source[0, :, :used_tokens, :]
                value_cache[row, :, :used_tokens, :] = value_source[0, :, :used_tokens, :]
            merged.append((key_cache, value_cache))
        return merged

    def _select_active_cache_rows(
        self,
        active_cache: list[tuple[torch.Tensor, torch.Tensor]],
        keep_positions: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        keep_tensor = torch.tensor(keep_positions, dtype=torch.long, device=self.device)
        selected: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key_cache, value_cache in active_cache:
            selected.append(
                (
                    key_cache.index_select(0, keep_tensor).contiguous(),
                    value_cache.index_select(0, keep_tensor).contiguous(),
                )
            )
        return selected

    def _next_token(self, hidden_states: torch.Tensor) -> int:
        return int(self._next_tokens(hidden_states).item())

    def _next_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden_states[:, -1, :], self._weight("model.embed_tokens.weight"))
        return torch.argmax(logits, dim=-1)

    def _decode_step_batch(
        self,
        next_tokens: torch.Tensor,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        input_ids = next_tokens.to(device=self.device, dtype=torch.long).view(-1, 1)
        positions = lengths.to(device=self.device, dtype=torch.long).view(-1, 1)
        cos, sin = self._rotary_cos_sin(positions)
        hidden_states = self._weight("model.embed_tokens.weight")[input_ids]

        for layer_index in range(self.num_layers):
            residual = hidden_states
            hidden_states = _rms_norm(
                hidden_states,
                self._weight(f"model.layers.{layer_index}.input_layernorm.weight"),
                self.rms_eps,
            )
            attn_output = self._self_attention_batch_decode(
                hidden_states,
                layer_index,
                cos,
                sin,
                cache[layer_index],
                lengths,
            )
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = _rms_norm(
                hidden_states,
                self._weight(f"model.layers.{layer_index}.post_attention_layernorm.weight"),
                self.rms_eps,
            )
            hidden_states = residual + self._mlp(hidden_states, layer_index)

        hidden_states = _rms_norm(hidden_states, self._weight("model.norm.weight"), self.rms_eps)
        return hidden_states, cache, lengths + 1

    @torch.inference_mode()
    def _generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        return self._generate_batch_tokenized(
            [self._tokenize(prompt) for prompt in prompts],
            max_new_tokens,
        )

    def _generate_length_aware(
        self,
        prompts: list[str],
        max_new_tokens: int,
        preferred_batch_size: int,
    ) -> list[str]:
        if not prompts:
            return []
        if max_new_tokens <= 0:
            return [""] * len(prompts)

        items = [
            (index, self._tokenize(prompt))
            for index, prompt in enumerate(prompts)
        ]
        preferred_batch_size = max(int(preferred_batch_size or 1), 1)
        if preferred_batch_size <= 1:
            return [
                self._generate_single_tokenized(token_ids, max_new_tokens)
                for _, token_ids in items
            ]

        outputs = [""] * len(items)
        sorted_items = sorted(items, key=lambda item: len(item[1]))
        current: list[tuple[int, list[int]]] = []

        def flush() -> None:
            if not current:
                return
            generated = self._generate_batch_tokenized(
                [token_ids for _, token_ids in current],
                max_new_tokens,
            )
            for (original_index, _), generated_text in zip(current, generated):
                outputs[original_index] = generated_text
            current.clear()

        for item in sorted_items:
            proposed = current + [item]
            lengths = [max(len(token_ids), 1) for _, token_ids in proposed]
            length_ratio = max(lengths) / max(min(lengths), 1)
            if (
                current
                and (
                    len(proposed) > preferred_batch_size
                    or length_ratio > self.batch_length_ratio
                )
            ):
                flush()
            current.append(item)
        flush()
        return outputs

    def _generate_batch_tokenized(
        self,
        tokenized: list[list[int]],
        max_new_tokens: int,
    ) -> list[str]:
        if not tokenized:
            return []
        if max_new_tokens <= 0:
            return [""] * len(tokenized)
        if len(tokenized) == 1:
            return [self._generate_single_tokenized(tokenized[0], max_new_tokens)]

        prompt_lengths = [len(token_ids) for token_ids in tokenized]
        total_capacity = max(prompt_lengths) + int(max_new_tokens)
        self._prepare_batch_prefix_cache(tokenized)

        last_hidden_states: list[torch.Tensor] = []
        single_caches: list[list[tuple[torch.Tensor, torch.Tensor, int]]] = []
        for token_ids in tokenized:
            hidden_states, cache = self._prefill_single(token_ids, total_capacity)
            last_hidden_states.append(hidden_states[:, -1:, :])
            single_caches.append(cache)

        cache = self._merge_caches(single_caches, total_capacity)
        lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=self.device)
        next_tokens = self._next_tokens(torch.cat(last_hidden_states, dim=0))
        generated_ids = [[int(token_id)] for token_id in next_tokens.tolist()]

        for _ in range(max_new_tokens - 1):
            hidden_states, cache, lengths = self._decode_step_batch(next_tokens, cache, lengths)
            next_tokens = self._next_tokens(hidden_states)
            for output_ids, token_id in zip(generated_ids, next_tokens.tolist()):
                output_ids.append(int(token_id))

        return [
            self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for output_ids in generated_ids
        ]

    def _generate_single(self, prompt: str, max_new_tokens: int) -> str:
        return self._generate_single_tokenized(self._tokenize(prompt), max_new_tokens)

    def _generate_single_tokenized(self, token_ids: list[int], max_new_tokens: int) -> str:
        if max_new_tokens <= 0:
            return ""

        hidden_states, cache = self._prefill_single(token_ids, len(token_ids) + max_new_tokens)
        next_token = self._next_token(hidden_states)
        generated_ids = [next_token]

        current_position = len(token_ids)
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

    @torch.inference_mode()
    def _generate_one(self, prompt: str, max_new_tokens: int) -> str:
        return self._generate_single(prompt, max_new_tokens)
