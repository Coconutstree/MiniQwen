"""使用可手动编辑权重的最小 PyTorch Transformer 推理流程。"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


# =============================================================================
# 手动参数区：可直接修改以下文本、词表和权重后重新运行本文件
# =============================================================================

INPUT_TEXT = "中国首都是"

TOKEN_TO_ID = {
    "中": 10,
    "国": 20,
    "首": 30,
    "都": 40,
    "是": 50,
    "北": 60,
    "京": 70,
    "上": 80,
    "海": 90,
}

# LM Head 候选 token；字典顺序必须与 W_LM 的行顺序一致。
CANDIDATE_TOKENS = {
    60: "北",
    70: "京",
    80: "上",
    90: "海",
}

EMBEDDING_TABLE = {
    10: [2.0, 0.0, 0.0, 0.0],
    20: [0.0, 2.0, 0.0, 0.0],
    30: [0.0, 0.0, 2.0, 0.0],
    40: [0.0, 0.0, 0.0, 2.0],
    50: [1.0, 1.0, -1.0, -1.0],
    60: [1.0, 1.0, 0.0, 0.0],
    70: [0.8, 0.8, 0.0, 0.0],
    80: [0.0, 0.0, 1.0, 1.0],
    90: [0.0, 0.0, 0.8, 0.8],
}

GAMMA_ATTN = [1.0, 1.0, 1.0, 1.0]
GAMMA_FFN = [1.0, 1.0, 1.0, 1.0]

WQ = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
WK = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
WV = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
WO = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

W_GATE = [
    [-0.2252, -0.2305, -0.0501, -0.0868, 0.1697, 0.1384, -0.0632, -0.4230],
    [0.0645, -0.2527, 0.0700, 0.0616, 0.0240, 0.2475, 0.2234, -0.0495],
    [-0.2705, -0.3392, 0.1133, 0.1587, 0.1198, -0.3110, -0.0683, 0.3706],
    [0.1500, -0.1171, -0.0347, 0.0367, 0.2779, 0.3173, 0.1893, -0.1687],
]
W_UP = [
    [-0.1227, 0.0063, -0.0985, 0.0497, 0.0879, 0.0225, 0.1282, 0.0882],
    [-0.0205, 0.1585, -0.0579, 0.0105, 0.1046, 0.4604, -0.2938, -0.3173],
    [-0.1346, 0.1746, 0.2111, 0.0356, -0.0461, -0.0784, 0.1087, -0.0790],
    [-0.0892, 0.1488, 0.3042, 0.6821, -0.3062, -0.2468, 0.3639, -0.1103],
]
W_DOWN = [
    [-0.1138, 0.1840, 0.2222, 0.2580],
    [-0.2956, 0.5135, -0.0946, 0.0671],
    [-0.3259, -0.1100, -0.0960, -0.0999],
    [-0.2134, 0.2230, -0.0281, 0.1612],
    [-0.0187, 0.1374, -0.1677, 0.0002],
    [0.1684, -0.0800, 0.2079, 0.0716],
    [-0.0492, 0.4605, -0.3763, -0.0099],
    [-0.2090, -0.1913, 0.0067, 0.1420],
]

W_LM = [
    [2.0, 2.0, -1.0, -1.0],  # 北
    [1.2, 1.2, -0.5, -0.5],  # 京
    [-1.0, -1.0, 2.0, 2.0],  # 上
    [-0.8, -0.8, 1.2, 1.2],  # 海
]

# =============================================================================
# 推理函数
# =============================================================================


def simple_tokenizer(
    text: str, token_to_id: Mapping[str, int] = TOKEN_TO_ID
) -> torch.Tensor:
    """使用固定白板词表把文本转换为 token IDs。"""
    if not text:
        raise ValueError("输入文本不能为空")

    token_ids = []
    for position, character in enumerate(text):
        if character not in token_to_id:
            raise ValueError(f"未知字符 {character!r}，位置 {position}")
        token_ids.append(token_to_id[character])
    return torch.tensor(token_ids, dtype=torch.long)


def embedding_lookup(
    token_ids: torch.Tensor,
    embedding_table: Mapping[int, Sequence[float] | torch.Tensor],
) -> torch.Tensor:
    """按非连续 token ID 从 Embedding 表中取向量。"""
    if token_ids.dtype != torch.long or token_ids.ndim != 1:
        raise TypeError("token_ids 必须是一维 torch.long 张量")

    rows = []
    for token_id in token_ids.tolist():
        if token_id not in embedding_table:
            raise KeyError(f"Embedding 表中不存在 token_id={token_id}")
        rows.append(torch.as_tensor(embedding_table[token_id], dtype=torch.float32))
    return torch.stack(rows)


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6
) -> torch.Tensor:
    """沿最后一维执行 RMSNorm。"""
    normalized = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normalized if weight is None else normalized * weight


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对形状为 [seq_len, head_dim] 的 Q、K 应用 RoPE。"""
    if q.shape != k.shape or q.ndim != 2:
        raise ValueError("q 和 k 必须具有相同的 [seq_len, head_dim] shape")
    head_dim = q.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE 要求 head_dim 为偶数")

    positions = torch.arange(q.shape[0], dtype=q.dtype, device=q.device)
    dimension_indices = torch.arange(0, head_dim, 2, dtype=q.dtype, device=q.device)
    inverse_frequencies = 1.0 / (10000.0 ** (dimension_indices / head_dim))
    angles = torch.outer(positions, inverse_frequencies)
    cosines, sines = angles.cos(), angles.sin()

    def rotate(x: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        pairs = torch.stack(
            (even * cosines - odd * sines, even * sines + odd * cosines), dim=-1
        )
        return pairs.flatten(start_dim=-2)

    return rotate(q), rotate(k)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """执行单头缩放因果自注意力，返回输出和注意力权重。"""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 2:
        raise ValueError("q、k、v 必须具有相同的 [seq_len, head_dim] shape")

    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    causal_mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights @ v, attention_weights


def toy_ffn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """执行无 bias 的 gated SiLU FFN。"""
    return (F.silu(x @ w_gate) * (x @ w_up)) @ w_down


def lm_head(last_hidden: torch.Tensor, w_lm: torch.Tensor) -> torch.Tensor:
    """使用 [候选词数, hidden_size] 权重计算候选 token logits。"""
    return last_hidden @ w_lm.transpose(0, 1)


def _tensor(name: str, values: Sequence[object], expected_shape: tuple[int, ...]) -> torch.Tensor:
    """把手动参数转为 float32 张量，并提供清晰的 shape 错误。"""
    try:
        tensor = torch.tensor(values, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"参数 {name} 不是规则数值数组: {error}") from error
    actual_shape = tuple(tensor.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"参数 {name} shape 错误：期望 {expected_shape}，实际 {actual_shape}"
        )
    return tensor


def _print_tensor(name: str, value: torch.Tensor) -> None:
    print(f"{name}: shape={tuple(value.shape)}")
    print(value)


def main() -> None:
    torch.set_printoptions(precision=6, sci_mode=False)
    hidden_size = len(GAMMA_ATTN)
    ffn_size = len(W_DOWN)
    candidate_ids = list(CANDIDATE_TOKENS)

    # 将顶部手动参数转换为张量，并在计算前校验所有 shape。
    gamma_attn = _tensor("GAMMA_ATTN", GAMMA_ATTN, (hidden_size,))
    gamma_ffn = _tensor("GAMMA_FFN", GAMMA_FFN, (hidden_size,))
    wq = _tensor("WQ", WQ, (hidden_size, hidden_size))
    wk = _tensor("WK", WK, (hidden_size, hidden_size))
    wv = _tensor("WV", WV, (hidden_size, hidden_size))
    wo = _tensor("WO", WO, (hidden_size, hidden_size))
    w_gate = _tensor("W_GATE", W_GATE, (hidden_size, ffn_size))
    w_up = _tensor("W_UP", W_UP, (hidden_size, ffn_size))
    w_down = _tensor("W_DOWN", W_DOWN, (ffn_size, hidden_size))
    w_lm = _tensor("W_LM", W_LM, (len(candidate_ids), hidden_size))

    for token_id, vector in EMBEDDING_TABLE.items():
        actual_size = len(vector)
        if actual_size != hidden_size:
            raise ValueError(
                f"参数 EMBEDDING_TABLE[{token_id}] shape 错误："
                f"期望 ({hidden_size},)，实际 ({actual_size},)"
            )

    print(f"PyTorch version: {torch.__version__}")
    print("Device: cpu")

    # 1. Tokenizer
    token_ids = simple_tokenizer(INPUT_TEXT)
    print(f"输入文本: {INPUT_TEXT}")
    print(f"token_ids: {token_ids.tolist()} | shape={tuple(token_ids.shape)}")

    # 2. Embedding
    x = embedding_lookup(token_ids, EMBEDDING_TABLE)
    _print_tensor("Embedding X", x)

    # 3-4. Attention 前 RMSNorm 和显式 Q/K/V 矩阵乘法
    normalized_x = rms_norm(x, gamma_attn)
    q, k, v = normalized_x @ wq, normalized_x @ wk, normalized_x @ wv
    _print_tensor("Attention RMSNorm", normalized_x)
    print(f"Q/K/V shapes: {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}")

    # 5. RoPE
    q, k = apply_rope(q, k)
    print(f"RoPE Q/K shapes: {tuple(q.shape)}, {tuple(k.shape)}")

    # 6-7. 因果 Attention、输出投影和第一次残差连接
    attention, attention_weights = scaled_dot_product_attention(q, k, v)
    attention_output = attention @ wo
    x_mid = x + attention_output
    _print_tensor("Attention weights", attention_weights)
    _print_tensor("Attention output", attention_output)
    _print_tensor("第一次残差 Xmid", x_mid)
    assert torch.count_nonzero(attention_weights.triu(diagonal=1)).item() == 0

    # 8. FFN 前 RMSNorm、FFN 和第二次残差连接
    normalized_x_mid = rms_norm(x_mid, gamma_ffn)
    mlp_output = toy_ffn(normalized_x_mid, w_gate, w_up, w_down)
    x_out = x_mid + mlp_output
    _print_tensor("FFN RMSNorm", normalized_x_mid)
    _print_tensor("MLP output", mlp_output)
    _print_tensor("第二次残差 Xout", x_out)

    # 9. 使用最后一个 token 的 hidden state 预测下一个 token
    logits = lm_head(x_out[-1], w_lm)
    probabilities = torch.softmax(logits, dim=-1)
    predicted_index = int(torch.argmax(logits).item())
    predicted_id = candidate_ids[predicted_index]
    predicted_token = CANDIDATE_TOKENS[predicted_id]
    _print_tensor("LM Head logits", logits)
    _print_tensor("LM Head probabilities", probabilities)
    print(f"预测出的下一个 token_id = {predicted_id}")
    print(f"预测出的下一个 token = {predicted_token}")
    print(f"拼接结果 = {INPUT_TEXT + predicted_token}")


if __name__ == "__main__":
    main()
