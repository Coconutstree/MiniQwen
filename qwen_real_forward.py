"""Manual Qwen2.5-0.5B-Instruct forward pass with Transformers alignment.

This script loads a local Qwen model package, extracts tokenizer/config/weights
from a Transformers model instance, manually computes input_ids -> logits, and
compares every decoder layer plus final logits against the official forward.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_PROMPT = "中国首都是"
DEFAULT_MODEL_PATH = "model/Qwen2.5-0.5B-Instruct"


torch = None
F = None
AutoModelForCausalLM = None
AutoTokenizer = None


@dataclass(frozen=True)
class RuntimeDeps:
    torch: Any
    functional: Any
    auto_model_for_causal_lm: Any
    auto_tokenizer: Any


@dataclass(frozen=True)
class ForwardTrace:
    logits: Any
    final_hidden_states: Any
    layer_hidden_states: list[Any]


@dataclass(frozen=True)
class ShapeSnapshot:
    layer_index: int
    norm_shape: tuple[int, ...]
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    repeated_k_shape: tuple[int, ...]
    repeated_v_shape: tuple[int, ...]
    attention_output_shape: tuple[int, ...]
    hidden_shape: tuple[int, ...]


def require_runtime_deps() -> RuntimeDeps:
    """Import heavy runtime dependencies only when execution really needs them."""
    missing: list[str] = []

    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
    except ModuleNotFoundError:
        missing.append("torch")
        torch_module = None
        functional_module = None

    try:
        from transformers import AutoModelForCausalLM as model_cls
        from transformers import AutoTokenizer as tokenizer_cls
    except ModuleNotFoundError:
        missing.append("transformers")
        model_cls = None
        tokenizer_cls = None

    if importlib.util.find_spec("safetensors") is None:
        missing.append("safetensors")

    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise SystemExit(
            f"Missing runtime dependencies: {joined}\n"
            "Install PyTorch for your CPU/CUDA environment, then install the "
            "packages listed in data/student_release/requirements.txt."
        )

    globals()["torch"] = torch_module
    globals()["F"] = functional_module
    globals()["AutoModelForCausalLM"] = model_cls
    globals()["AutoTokenizer"] = tokenizer_cls
    return RuntimeDeps(torch_module, functional_module, model_cls, tokenizer_cls)


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_model_path(value: str | None) -> str:
    return value or os.environ.get("HF_MODEL_NAME") or DEFAULT_MODEL_PATH


def resolve_device(value: str | None) -> str:
    if value:
        return value
    env_device = os.environ.get("HF_DEVICE")
    if env_device:
        return env_device
    deps = require_runtime_deps()
    return "cuda" if deps.torch.cuda.is_available() else "cpu"


def torch_dtype_from_name(name: str) -> Any:
    normalized = name.lower()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported dtype {name!r}. Use float32, float16, bfloat16, or auto."
        )
    return mapping[normalized]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a manual Qwen2.5-0.5B-Instruct forward and compare it "
        "with Transformers."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Local model directory or HF id. Defaults to HF_MODEL_NAME or {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, or a device like cuda:0. Defaults to HF_DEVICE or auto.",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("HF_DTYPE", "float32"),
        help="float32, float16, bfloat16, or auto. Defaults to float32.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=parse_bool_env("HF_LOCAL_FILES_ONLY", False),
        help="Only read local model files. Also enabled by HF_LOCAL_FILES_ONLY=1.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt text. Defaults to {DEFAULT_PROMPT!r}.",
    )
    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="Tokenize prompt directly instead of using the Qwen chat template.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Override relative allclose tolerance.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Override absolute allclose tolerance.",
    )
    parser.add_argument(
        "--quiet-shapes",
        action="store_true",
        help="Suppress per-layer shape logs.",
    )
    return parser


def load_qwen_model(
    model_path: str,
    device: str,
    dtype: Any,
    local_files_only: bool,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    model.to(device)
    return tokenizer, model


def make_input_ids(
    tokenizer: Any,
    prompt: str,
    device: str,
    plain_text: bool,
) -> tuple[str, Any]:
    if plain_text:
        text = prompt
    else:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    encoded = tokenizer([text], return_tensors="pt")
    return text, encoded.input_ids.to(device)


def rms_norm(x: Any, weight: Any, eps: float) -> Any:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps)).to(input_dtype)


def rotate_half(x: Any) -> Any:
    half = x.shape[-1] // 2
    first_half = x[..., :half]
    second_half = x[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)


def build_rotary_cos_sin(
    position_ids: Any,
    head_dim: int,
    rope_theta: float,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any]:
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    position_ids = position_ids.to(device=device, dtype=torch.float32)
    freqs = torch.einsum("bi,j->bij", position_ids, inv_freq)
    embeddings = torch.cat((freqs, freqs), dim=-1)
    return embeddings.cos().to(dtype=dtype), embeddings.sin().to(dtype=dtype)


def get_rope_theta(config: Any) -> float:
    if hasattr(config, "rope_theta"):
        return float(config.rope_theta)

    for field_name in ("rope_scaling", "rope_parameters"):
        rope_config = getattr(config, field_name, None)
        if isinstance(rope_config, dict) and "rope_theta" in rope_config:
            return float(rope_config["rope_theta"])

    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    for field_name in ("rope_scaling", "rope_parameters"):
        rope_config = config_dict.get(field_name)
        if isinstance(rope_config, dict) and "rope_theta" in rope_config:
            return float(rope_config["rope_theta"])

    return 10000.0


def apply_qwen_rope(q: Any, k: Any, cos: Any, sin: Any) -> tuple[Any, Any]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: Any, n_rep: int) -> Any:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def make_causal_mask(batch_size: int, seq_len: int, dtype: Any, device: Any) -> Any:
    min_value = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_value, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)


def qwen_attention(
    hidden_states: Any,
    layer: Any,
    config: Any,
    position_ids: Any,
) -> tuple[Any, ShapeSnapshot]:
    batch_size, seq_len, _ = hidden_states.shape
    num_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // num_heads
    num_key_value_groups = num_heads // num_key_value_heads

    query_states = F.linear(
        hidden_states,
        layer.self_attn.q_proj.weight,
        layer.self_attn.q_proj.bias,
    )
    key_states = F.linear(
        hidden_states,
        layer.self_attn.k_proj.weight,
        layer.self_attn.k_proj.bias,
    )
    value_states = F.linear(
        hidden_states,
        layer.self_attn.v_proj.weight,
        layer.self_attn.v_proj.bias,
    )

    query_states = query_states.view(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        batch_size,
        seq_len,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        batch_size,
        seq_len,
        num_key_value_heads,
        head_dim,
    ).transpose(1, 2)

    cos, sin = build_rotary_cos_sin(
        position_ids,
        head_dim=head_dim,
        rope_theta=get_rope_theta(config),
        dtype=query_states.dtype,
        device=query_states.device,
    )
    query_states, key_states = apply_qwen_rope(query_states, key_states, cos, sin)

    repeated_key_states = repeat_kv(key_states, num_key_value_groups)
    repeated_value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(
        query_states,
        repeated_key_states.transpose(2, 3),
    ) / math.sqrt(head_dim)
    attn_weights = attn_weights + make_causal_mask(
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=attn_weights.dtype,
        device=attn_weights.device,
    )
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, repeated_value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_len, int(config.hidden_size))
    attn_output = F.linear(attn_output, layer.self_attn.o_proj.weight, None)

    snapshot = ShapeSnapshot(
        layer_index=-1,
        norm_shape=tuple(hidden_states.shape),
        q_shape=tuple(query_states.shape),
        k_shape=tuple(key_states.shape),
        v_shape=tuple(value_states.shape),
        repeated_k_shape=tuple(repeated_key_states.shape),
        repeated_v_shape=tuple(repeated_value_states.shape),
        attention_output_shape=tuple(attn_output.shape),
        hidden_shape=tuple(attn_output.shape),
    )
    return attn_output, snapshot


def qwen_mlp(hidden_states: Any, layer: Any) -> Any:
    gate = F.linear(hidden_states, layer.mlp.gate_proj.weight, None)
    up = F.linear(hidden_states, layer.mlp.up_proj.weight, None)
    return F.linear(F.silu(gate) * up, layer.mlp.down_proj.weight, None)


def manual_forward(
    model: Any,
    input_ids: Any,
    print_shapes: bool,
) -> ForwardTrace:
    config = model.config
    batch_size, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    position_ids = position_ids.expand(batch_size, seq_len)

    hidden_states = model.model.embed_tokens.weight[input_ids]
    layer_hidden_states: list[Any] = []

    if print_shapes:
        print(f"input_ids: shape={tuple(input_ids.shape)}")
        print(f"embedding hidden_states: shape={tuple(hidden_states.shape)}")

    for layer_index, layer in enumerate(model.model.layers):
        residual = hidden_states
        normalized = rms_norm(
            hidden_states,
            layer.input_layernorm.weight,
            eps=float(config.rms_norm_eps),
        )
        attn_output, shape_snapshot = qwen_attention(
            hidden_states=normalized,
            layer=layer,
            config=config,
            position_ids=position_ids,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        normalized = rms_norm(
            hidden_states,
            layer.post_attention_layernorm.weight,
            eps=float(config.rms_norm_eps),
        )
        hidden_states = residual + qwen_mlp(normalized, layer)
        layer_hidden_states.append(hidden_states.detach().float().cpu())

        if print_shapes:
            snapshot = ShapeSnapshot(
                layer_index=layer_index,
                norm_shape=shape_snapshot.norm_shape,
                q_shape=shape_snapshot.q_shape,
                k_shape=shape_snapshot.k_shape,
                v_shape=shape_snapshot.v_shape,
                repeated_k_shape=shape_snapshot.repeated_k_shape,
                repeated_v_shape=shape_snapshot.repeated_v_shape,
                attention_output_shape=shape_snapshot.attention_output_shape,
                hidden_shape=tuple(hidden_states.shape),
            )
            print_layer_shapes(snapshot)

    final_hidden_states = rms_norm(
        hidden_states,
        model.model.norm.weight,
        eps=float(config.rms_norm_eps),
    )
    logits = F.linear(final_hidden_states, model.model.embed_tokens.weight, None)

    if print_shapes:
        print(f"final hidden_states: shape={tuple(final_hidden_states.shape)}")
        print(f"manual logits: shape={tuple(logits.shape)}")

    return ForwardTrace(
        logits=logits,
        final_hidden_states=final_hidden_states.detach().float().cpu(),
        layer_hidden_states=layer_hidden_states,
    )


def print_layer_shapes(snapshot: ShapeSnapshot) -> None:
    print(
        "layer "
        f"{snapshot.layer_index:02d}: "
        f"norm={snapshot.norm_shape}, "
        f"q={snapshot.q_shape}, "
        f"k={snapshot.k_shape}, "
        f"v={snapshot.v_shape}, "
        f"repeat_k={snapshot.repeated_k_shape}, "
        f"repeat_v={snapshot.repeated_v_shape}, "
        f"attn_out={snapshot.attention_output_shape}, "
        f"hidden={snapshot.hidden_shape}"
    )


def extract_first_tensor(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def official_forward_with_hooks(model: Any, input_ids: Any) -> ForwardTrace:
    layer_hidden_states: list[Any | None] = [None] * int(model.config.num_hidden_layers)
    final_hidden_states: dict[str, Any] = {}
    handles = []

    def make_layer_hook(layer_index: int) -> Callable[[Any, Any, Any], None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            layer_hidden_states[layer_index] = (
                extract_first_tensor(output).detach().float().cpu()
            )

        return hook

    def final_norm_hook(_module: Any, _inputs: Any, output: Any) -> None:
        final_hidden_states["value"] = extract_first_tensor(output).detach().float().cpu()

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_layer_hook(layer_index)))
    handles.append(model.model.norm.register_forward_hook(final_norm_hook))

    try:
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones(
            (batch_size, seq_len),
            dtype=torch.long,
            device=input_ids.device,
        )
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing_layers = [
        index for index, hidden in enumerate(layer_hidden_states) if hidden is None
    ]
    if missing_layers:
        raise RuntimeError(f"Official layer hooks did not fire: {missing_layers}")
    if "value" not in final_hidden_states:
        raise RuntimeError("Official final norm hook did not fire.")

    return ForwardTrace(
        logits=outputs.logits,
        final_hidden_states=final_hidden_states["value"],
        layer_hidden_states=[hidden for hidden in layer_hidden_states if hidden is not None],
    )


def compare_tensors(name: str, manual: Any, official: Any, rtol: float, atol: float) -> bool:
    manual_cpu = manual.detach().float().cpu() if hasattr(manual, "detach") else manual
    official_cpu = official.detach().float().cpu() if hasattr(official, "detach") else official
    if tuple(manual_cpu.shape) != tuple(official_cpu.shape):
        print(
            f"{name}: shape mismatch manual={tuple(manual_cpu.shape)} "
            f"official={tuple(official_cpu.shape)}"
        )
        return False

    diff = (manual_cpu - official_cpu).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    allclose = torch.allclose(manual_cpu, official_cpu, rtol=rtol, atol=atol)
    print(
        f"{name}: shape={tuple(manual_cpu.shape)}, "
        f"max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}, allclose={allclose}"
    )
    return bool(allclose)


def apply_repetition_penalty(logits: Any, input_ids: Any, penalty: float) -> Any:
    if penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for batch_index in range(input_ids.shape[0]):
        previous_token_ids = torch.unique(input_ids[batch_index])
        previous_scores = adjusted[batch_index, previous_token_ids]
        adjusted[batch_index, previous_token_ids] = torch.where(
            previous_scores < 0,
            previous_scores * penalty,
            previous_scores / penalty,
        )
    return adjusted


def print_next_token(label: str, logits: Any, tokenizer: Any) -> int:
    next_token_id = int(torch.argmax(logits, dim=-1).item())
    next_token = tokenizer.decode([next_token_id])
    print(f"{label} next token id: {next_token_id}")
    print(f"{label} next token: {next_token!r}")
    return next_token_id


def default_tolerances(dtype_name: str) -> tuple[float, float]:
    normalized = dtype_name.lower()
    if normalized in {"float16", "fp16", "half", "bfloat16", "bf16", "auto"}:
        return 1e-2, 1e-2
    return 1e-5, 1e-5


def print_model_summary(model: Any, model_path: str, device: str, dtype_name: str) -> None:
    config = model.config
    embed_weight = model.model.embed_tokens.weight
    print(f"model_path: {model_path}")
    print(f"device: {device}")
    print(f"dtype: {dtype_name}")
    print(f"architecture: {getattr(config, 'architectures', ['unknown'])[0]}")
    print(
        "config: "
        f"layers={config.num_hidden_layers}, "
        f"hidden_size={config.hidden_size}, "
        f"heads={config.num_attention_heads}, "
        f"kv_heads={config.num_key_value_heads}, "
        f"head_dim={config.hidden_size // config.num_attention_heads}, "
        f"intermediate_size={config.intermediate_size}, "
        f"vocab_size={config.vocab_size}, "
        f"rope_theta={get_rope_theta(config)}, "
        f"rms_norm_eps={config.rms_norm_eps}"
    )
    print(f"embed_tokens.weight: shape={tuple(embed_weight.shape)}, dtype={embed_weight.dtype}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    deps = require_runtime_deps()

    model_path = resolve_model_path(args.model_path)
    device = resolve_device(args.device)
    dtype = torch_dtype_from_name(args.dtype)
    rtol, atol = default_tolerances(args.dtype)
    if args.rtol is not None:
        rtol = args.rtol
    if args.atol is not None:
        atol = args.atol

    path = Path(model_path)
    if path.exists() and not path.is_dir():
        raise SystemExit(f"Model path exists but is not a directory: {model_path}")

    print(f"PyTorch version: {deps.torch.__version__}")
    tokenizer, model = load_qwen_model(
        model_path=model_path,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    print_model_summary(model, model_path, device, args.dtype)

    rendered_text, input_ids = make_input_ids(
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        plain_text=args.plain_text,
    )
    print(f"prompt: {args.prompt!r}")
    print(f"tokenization mode: {'plain text' if args.plain_text else 'chat template'}")
    print(f"rendered text length: {len(rendered_text)} characters")
    print(f"input_ids: {input_ids.tolist()} | shape={tuple(input_ids.shape)}")

    with torch.inference_mode():
        manual_trace = manual_forward(
            model=model,
            input_ids=input_ids,
            print_shapes=not args.quiet_shapes,
        )
        official_trace = official_forward_with_hooks(model=model, input_ids=input_ids)

    print("\nAlignment report")
    all_ok = True
    for layer_index, (manual_hidden, official_hidden) in enumerate(
        zip(manual_trace.layer_hidden_states, official_trace.layer_hidden_states)
    ):
        all_ok = (
            compare_tensors(
                f"layer_{layer_index:02d}_hidden",
                manual_hidden,
                official_hidden,
                rtol=rtol,
                atol=atol,
            )
            and all_ok
        )

    all_ok = (
        compare_tensors(
            "final_norm_hidden",
            manual_trace.final_hidden_states,
            official_trace.final_hidden_states,
            rtol=rtol,
            atol=atol,
        )
        and all_ok
    )
    all_ok = (
        compare_tensors(
            "logits",
            manual_trace.logits,
            official_trace.logits,
            rtol=rtol,
            atol=atol,
        )
        and all_ok
    )

    raw_next_logits = manual_trace.logits[:, -1, :]
    print_next_token("raw forward", raw_next_logits, tokenizer)

    repetition_penalty = float(
        getattr(model.generation_config, "repetition_penalty", 1.0) or 1.0
    )
    if repetition_penalty != 1.0:
        adjusted_next_logits = apply_repetition_penalty(
            raw_next_logits,
            input_ids=input_ids,
            penalty=repetition_penalty,
        )
        print(f"generation_config repetition_penalty: {repetition_penalty}")
        print_next_token("generation-adjusted", adjusted_next_logits, tokenizer)

    print(f"all comparisons within tolerance: {all_ok}")


if __name__ == "__main__":
    main()
