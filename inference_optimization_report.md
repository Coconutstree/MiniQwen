# MiniQwen 推理优化实现报告

报告日期：2026-07-18

## 1. 项目概述

本项目实现了一个面向 Qwen2.5-0.5B-Instruct 的手写推理引擎，核心入口为 `student_release/student_engine.py` 中的 `StudentEngine`。实现目标是在不调用 Hugging Face `AutoModelForCausalLM`、`model.generate()`、vLLM、llama.cpp 等完整推理框架的前提下，完成模型权重加载、forward 计算、prefill、decode、KV cache、批处理和 serving 调度。

当前实现具备以下特点：

- 只使用 `AutoTokenizer` 处理 tokenizer 和 chat template；
- 直接读取本地 `config.json` 与 safetensors 权重；
- 手写 Qwen2 decoder block，包括 RMSNorm、RoPE、GQA Attention、MLP、LM Head；
- 使用 greedy decode，输出只包含 continuation；
- 支持固定 batch 推理与 `serve_requests()` 请求流接口；
- 对 decode 阶段实现 KV cache 复用、cache 预分配、批量 decode 和长度感知分组。

## 2. 运行环境配置

真实运行与评测环境按以下资源配置理解：

| 配置项 | 取值 |
| --- | --- |
| 加速设备 | GPU |
| GPU 型号 | NVIDIA GeForce RTX 2080 Ti |
| CPU 资源 | 4 核 |
| 可用资源约束 | 12G |
| 服务器 | `host=10-9-95-212` |
| 推理设备参数 | `device=cuda` |
| 计算精度 | `dtype=float16` |
| Attention 后端 | `attn_implementation=sdpa` |
| 模型来源 | 本地 `Qwen2.5-0.5B-Instruct` 目录，`local_files_only=True` |

在该配置下，推理主路径运行在 GPU 上，权重以 float16 放置到目标设备，attention 优先使用 PyTorch SDPA。代码中保留了 CUDA 不可用时的 CPU 兜底逻辑，但这属于本地调试兼容路径，不作为本报告的目标评测环境。

## 3. 文件依据

| 文件 | 作用 |
| --- | --- |
| `student_release/student_engine.py` | 学生推理引擎主实现，包含接口、模型 forward、KV cache、batch decode 和 serving 调度 |
| `student_release/utils/load_weights.py` | 读取本地模型目录、config 和 safetensors 权重 |
| `student_release/scripts/validate_engine.py` | 静态检查入口签名与禁用 API，防止调用完整模型或 benchmark 特判 |
| `student_release/scripts/run_inference_benchmark.py` | 评测脚本，包含 long context、decode、TTFT、serving、mixed、cache stress 等 suite |
| 最终实验终端输出 | 完整 GPU 评测结果，`limit=None`、`suite-isolation=process`、`timed-repeats=3` |

## 4. 采用接口

### 4.1 初始化接口

```python
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
        ...
```

初始化阶段完成以下工作：

- 固定随机种子，保证 greedy decode 的可复现性；
- 在真实评测环境中使用 `cuda` 作为推理设备；若本地调试机没有 CUDA，代码会自动回退到 CPU；
- 加载模型 config 与 safetensors 权重；
- 根据 `dtype` 和设备选择实际计算精度；GPU 环境使用 `float16`，CPU 兜底路径使用 `float32`；
- 加载 `AutoTokenizer`，并在缺少 pad token 时复用 eos token；
- 解析 Qwen2 结构参数：hidden size、层数、attention heads、KV heads、head dim、RMSNorm eps、RoPE theta；
- 预计算 RoPE 所需的 `inv_freq`。

### 4.2 生成接口

```python
def generate(
    self,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int = 1,
    suite_name: str | None = None,
) -> list[str]:
    ...
```

`generate()` 是 benchmark 的主入口。当前实现会忽略 `suite_name`，避免基于 suite 名称做特判。方法内部统一调用 `_generate_length_aware()`，根据 prompt token 长度进行分组，然后执行单样本或批量 greedy decode。

接口保证：

- 返回 `list[str]`；
- 输出数量与输入 prompt 数量一致；
- 输出顺序与输入顺序一致；
- 只返回新生成 continuation；
- 不进行随机采样；
- 当 `max_new_tokens <= 0` 时返回空 continuation。

### 4.3 Serving 调度接口

```python
def serve_requests(self, requests: list[dict], batch_size: int | None = None):
    ...
```

`serve_requests()` 面向 request stream 场景。当前实现使用 `max_new_tokens` 作为一级分组依据，将 token 预算相同的请求合并后调用 `_generate_length_aware()`。这样可以减少同一批次中 decode 步数不一致带来的空转，同时保留原始请求顺序。

当前调度策略：

- 按请求中的 `max_new_tokens` 分组；
- 每组内部根据 `batch_size` 或组大小确定有效 batch；
- 使用长度感知分组继续拆分；
- 生成后按原始 index 回填输出。

## 5. 具体方法

### 5.1 权重与配置加载

`student_release/utils/load_weights.py` 提供了轻量权重加载能力。它只做三件事：

- 解析本地模型目录或 Hugging Face cache snapshot；
- 读取 `config.json`；
- 读取单文件或分片 safetensors，合并为普通 `state_dict`。

`StudentEngine` 在初始化时将所有权重移动到目标设备和目标 dtype，并保存在 `self.weights` 中。模型计算过程中通过 `_weight()` 和 `_linear()` 访问权重，不依赖 Transformers 模型 forward。

### 5.2 Tokenize 与 Chat Template

`_tokenize()` 会先判断 prompt 是否已经包含 `<|im_start|>`。如果没有，则调用 tokenizer 的 chat template，将普通用户输入包装成 Qwen Instruct 格式，并添加 generation prompt。

这样做可以兼容两类输入：

- 已经由 benchmark 构造好的 chat prompt；
- 普通自然语言 prompt。

当 tokenizer 返回空 token 序列时，实现会优先回退到 BOS token；如果 BOS 不可用，则使用 EOS token，保证推理入口始终有至少一个 token。

### 5.3 手写 Qwen2 Forward

`_forward_tokens()` 是完整 decoder forward 的核心路径。每层计算流程如下：

1. 读取 token embedding；
2. 根据当前位置生成 RoPE 的 cos/sin；
3. 对 attention 输入做 RMSNorm；
4. 计算 Q/K/V projection；
5. reshape 为多头格式，并对 Q/K 应用 RoPE；
6. 按 GQA 规则把 KV heads repeat 到 attention heads；
7. prefill 阶段使用 causal attention，decode 阶段使用已有 KV cache；
8. attention 输出经过 `o_proj` 后与 residual 相加；
9. 对 MLP 输入做 RMSNorm；
10. 使用 Qwen2/SwiGLU 形式的 `silu(gate) * up -> down_proj`；
11. 最后一层后执行 final RMSNorm；
12. 使用 embedding weight 作为 tied LM head，通过 argmax 选取 greedy token。

Attention 优先使用 PyTorch `scaled_dot_product_attention`。当 `attn_implementation` 不是 `sdpa` 时，代码提供手写 matmul + softmax fallback。

### 5.4 Prefill 与 Decode

单样本推理路径 `_generate_single_tokenized()` 分为两个阶段：

- Prefill：一次性输入完整 prompt，得到最后 hidden state 和每层 KV cache；
- Decode：每次只输入上一步生成的 token，复用已有 KV cache，逐步生成后续 token。

当前实现不在遇到 EOS 时提前停止，而是按照 `max_new_tokens` 固定步数生成。这与评测说明中的 fixed-step decode 场景匹配，有利于 decode throughput、cache stress 等 suite 的稳定计分。

### 5.5 KV Cache 管理

KV cache 的基本结构为每层一组 `(key_cache, value_cache, used_tokens)`。优化点包括：

- Prefill 后保留每层完整 KV；
- Decode 时直接把新 token 的 K/V 写入已有 cache；
- `_reserve_cache()` 会提前扩容到 `prompt_length + max_new_tokens`，减少 decode 循环中的反复 `torch.cat()`；
- 当已有 cache 容量足够时直接复用 buffer；
- 当容量不足时再创建更大 buffer 并拷贝已用 token。

批量 decode 中，cache 会被 `_merge_caches()` 合并为 batch 维度的 KV buffer，形状类似：

```text
[batch, num_kv_heads, total_capacity, head_dim]
```

每个样本的有效长度不同，因此 `_batch_decode_attention()` 使用 `valid_lengths` 构造 mask，避免 attention 读取 padding 或未写入区域。

### 5.6 批量 Decode

`_generate_batch_tokenized()` 的批量路径如下：

1. 对每个 prompt 单独 prefill，避免不同长度 prompt 的 prefill padding 浪费；
2. 将每个样本最后一个 hidden state 拼成 batch；
3. 使用 LM head 得到每个样本的首个新 token；
4. 合并所有样本的 KV cache；
5. 在 decode 循环中用 `_decode_step_batch()` 对整个 batch 一次处理一个 token；
6. 每步更新所有样本的 cache 和当前位置；
7. 最后逐个 decode token id 为文本。

该策略的核心收益是：prefill 阶段避免长短 prompt 混合导致的大量 padding，decode 阶段又能利用 batch 维度提升吞吐。

### 5.7 长度感知分组

`_generate_length_aware()` 会先 token 化所有 prompt，再按 prompt 长度排序。构造 batch 时使用两个限制：

- batch 数量不超过 `preferred_batch_size`；
- 当前组内最长 prompt 与最短 prompt 的长度比例不超过 `self.batch_length_ratio`，当前阈值为 `1.6`。

当拟加入的新样本使 batch 过大或长度差异过大时，当前 batch 会先 flush。每个 batch 生成完成后按原始 index 回填，保证最终输出顺序不变。

该方法主要优化混合长度请求场景，减少同一批次内长短 prompt 差异导致的 attention mask 浪费和尾延迟放大。

## 6. 优化方式总结

| 优化点 | 具体实现 | 预期收益 |
| --- | --- | --- |
| 手写 forward | RMSNorm、RoPE、GQA、Attention、MLP、LM Head 均在 `student_engine.py` 实现 | 满足 strict track 要求，避免调用禁用模型 API |
| SDPA Attention | `attn_implementation == "sdpa"` 时调用 `F.scaled_dot_product_attention` | 利用 PyTorch 内置优化 attention kernel |
| Greedy Decode | LM head 后直接 `argmax` | 输出确定、无采样开销 |
| KV Cache | 每层保存 K/V，decode 只计算新 token | 避免每步重复计算历史 token |
| Cache 预分配 | `_reserve_cache()` 提前申请总容量 | 降低 decode 循环中的内存重新分配和拷贝 |
| 批量 Decode | 合并 cache 后每步批量处理多个请求 | 提升 decode throughput |
| 有效长度 mask | `_batch_decode_attention()` 使用 `valid_lengths` 屏蔽未写入位置 | 保证变长 batch 的 attention 正确性 |
| 长度感知 batching | 按 token 长度排序并限制组内长度比例 | 减少长短样本混批导致的无效计算 |
| Serving 分组 | `serve_requests()` 按 `max_new_tokens` 分组 | 减少 decode 步数不一致导致的等待 |
| GPU 主路径 | 真实环境使用 `cuda + float16 + sdpa` | 充分利用 GPU 计算与低精度吞吐 |
| 4 核 / 12G 资源适配 | 避免引入额外大框架，权重与 KV cache 由引擎直接管理 | 控制内存占用，适配课程限制资源 |
| CPU 兜底 | CUDA 不可用时自动回退 CPU，并将低精度 dtype 转为 float32 | 提升本地调试可运行性和稳定性 |

## 7. 实现效果

### 7.1 最终实验配置

最终实验使用完整评测命令运行：

```bash
uv run python -u scripts/run_inference_benchmark.py \
  --model /root/model/Qwen2.5-0.5B-Instruct \
  --local-files-only \
  --device cuda \
  --dtype float16 \
  --attn-implementation sdpa \
  --baseline-summary data/public_baseline_summary.json \
  --timed-repeats 3 \
  --suite-isolation process \
  --worker-timeout-s 1800 \
  --output-dir results/final_eval
```

评测采用 6 个独立 worker process 分别运行各 suite，`limit=None`，因此这是完整公开数据集上的最终实验结果。模型初始化耗时约 1.62s 到 1.67s，初始化显存分配约 950 MB，reserved 约 992 MB；全流程最高 peak 为 1348 MB，未发生 OOM。

### 7.2 最终综合评分

| 评分项 | 得分 | 关键指标 |
| --- | ---: | --- |
| Long Context Correctness | 28.50 / 30 | partial=0.950，exact=0.933 |
| Decode TPS | 18.78 / 25 | best_tps=59.4，speedup=0.26x，valid=1.000 |
| TTFT / Prefill Latency | 17.41 / 20 | avg=0.079s，p95=0.231s，buckets=4 |
| Serving / Scheduling | 8.75 / 15 | tps=52.4，p95=19.979s，iface=serve_requests |
| Runtime Robustness | 10.00 / 10 | runtime=1.000，valid=0.958 |
| Final Score | 83.44 / 100 | CAP=100，Realism Guard=OK |

最终分数为 83.44 / 100，这是 `limit=None` 下的完整 GPU 评测结果。真实性检查 `Realism Guard: OK`，说明吞吐和延迟没有触发异常保护；组件分层评分 `raw=83.44`、`before_cap=83.44`，没有被质量门槛或 cap 额外压分。

### 7.3 各 Suite 运行结果

| Suite | 数据规模 | 有效率 | TPS / Latency | Batch 信息 | Peak 显存 |
| --- | ---: | ---: | --- | --- | ---: |
| long_context | 15 cases | 0.933 | 45.3 TPS | batch_size=1 | 1169 MB |
| decode_throughput | 12 cases x 3 batch sizes | 1.000 | 58.5 TPS，best_tps=59.4 | best_bs=1 | 977 MB |
| ttft_prefill | 8 cases | 1.000 | avg latency=0.079s，best_tps=12.6 | best_bs=1 | 1346 MB |
| serving_schedule | 12 requests | 0.917 | 52.4 TPS | best_bs=12，serve_requests | 1348 MB |
| mixed_serving | 12 cases x 2 batch sizes | 0.917 | 49.0 TPS，best_tps=49.3 | best_bs=1 | 1276 MB |
| decode_cache_stress | 4 cases x 2 batch sizes x 3 lengths | 0.917 | 58.9 TPS | primary_bs=4 | 1052 MB |

结果表明：在 RTX 2080 Ti 和 12G 资源约束下，当前实现显存占用较低，最高 peak 约 1.35 GB，远低于 12G 限制。主要瓶颈不是显存，而是 Python 手写逐层 forward、prefill 调度和 serving 策略。

### 7.4 Decode Throughput 分析

最终 decode throughput 为 58.5 TPS，评分采用的 best TPS 为 59.4，speedup 为 0.26x，valid=1.000。该结果说明当前 KV cache 与 greedy decode 路径正确稳定，decode 输出质量没有成为扣分点。

与本地中间结果相比，GPU 最终环境下 decode TPS 明显提升，但 best batch size 为 1，说明当前批量 decode 在更大 batch 下未形成理想线性收益。原因主要包括：prefill 仍按单样本执行、cache 合并存在额外开销、`_repeat_kv()` 会带来 KV 展开成本，以及 Python 循环逐层调度限制了 GPU 利用率。

### 7.5 TTFT / Prefill 分析

最终 TTFT 得分为 17.41 / 20，平均 TTFT 为 0.079s，P95 为 0.231s，覆盖 4 个长度 bucket。分项得分为：

| 评分项 | 得分 |
| --- | ---: |
| bucket_avg | 10.30 / 12 |
| bucket_p95 | 5.11 / 6 |
| quality | 2.00 / 2 |

结论：TTFT 是当前表现较好的部分，说明单样本 prefill、SDPA attention 和 GPU float16 路径能够有效降低首 token 延迟。仍有提升空间的部分主要是长上下文 p95，可通过 prefix reuse、chunked prefill、batch prefill 和更高效的 attention kernel 继续压低。

### 7.6 Serving 调度分析

最终 serving 得分为 8.75 / 15，`serve_requests()` 已被评测调用，serving TPS 为 52.4，P95 latency 为 19.979s。相比 decode 和 TTFT，serving 分数偏低，说明当前请求流调度仍是主要短板。

当前策略按 `max_new_tokens` 分组，再交给长度感知 batch 生成，能保证接口可用和输出顺序正确，但还不是严格意义上的 continuous batching。它没有根据 `arrival_time_ms` 动态接入请求，也没有利用 `priority`、`group_id`、共享前缀和 stream size 做更细粒度调度，因此 p95 latency 仍有明显优化空间。

### 7.7 Cache 与内存表现

最终 cache/memory diagnostic 显示：

- cache TPS：58.7；
- cache speedup：0.51x；
- cache growth：0.00 MB / 100 tokens；
- peak extra：398 MB；
- cache peak saving：6.6%；
- 全流程最高 peak：1348 MB。

结论：KV cache 预分配和原地写入机制有效控制了 decode 过程中的动态增长，长 decode 压力下没有 OOM。由于整体 peak 远低于 12G，后续可以在显存预算内尝试更激进的 batch prefill、continuous batching 或 prefix KV 复用，用更多显存换取更高吞吐和更低尾延迟。

## 8. 与公开 baseline 的对比

公开 baseline 位于 `student_release/data/public_baseline_summary.json`，其主要指标包括：

- `long_context` TPS：44.67，平均延迟：1.71s；
- `decode_throughput` 总体 TPS：101.03，best batch TPS：224.76；
- `ttft_prefill` 平均延迟：0.035s，P95：0.063s；
- `serving_schedule` TPS：425.93，P95：2.46s；
- `decode_cache_stress` TPS：148.01。

当前实现相对 baseline 的优势主要在正确性和接口完整性：

- long context 最终 partial=0.950、exact=0.933，质量指标与公开 baseline 接近；
- decode throughput 的 valid=1.000，TTFT 的 valid=1.000，说明生成有效性和首 token 路径稳定；
- `serve_requests()` 已被 serving suite 调用，serving interface 计入 `serve_requests`；
- runtime robustness 得分为 10.00 / 10，最终整体 runtime=1.000、valid=0.958；
- 全流程 peak 显存约 1348 MB，远低于 12G 资源约束，为后续扩大 batch 或缓存 prefix 留出了空间。

当前差距主要在吞吐和延迟：

- decode best TPS 为 59.4，对应 baseline speedup=0.26x，说明 Python 手写逐层 forward 的 kernel 融合和调度能力仍弱于 vLLM baseline；
- TTFT 平均 0.079s，对应 baseline latency speedup=0.45x，P95 为 0.231s，对长上下文仍有优化空间；
- serving TPS 为 52.4，对应 baseline speedup=0.12x，P95 为 19.979s，说明 request stream 调度仍是主要短板；
- prefill 阶段仍是逐 prompt 计算，没有 prefix reuse 或 chunked prefill；
- serving 调度还不是真正 continuous batching；
- batch prefill 未做 padding batch SDPA，而是单样本 prefill 后合并 cache，正确性较稳但初始化吞吐有限。

## 9. 局限与后续优化方向

后续若继续优化，优先级建议如下：

1. 实现真正的 continuous batching：按 arrival time 动态加入请求，在 decode 循环中维护 active set，而不是一次性分组后整批生成。
2. 引入 prefix reuse：对共享 system prompt、模板前缀或相同上下文片段缓存 prefill KV，降低 TTFT 与 serving p95。
3. 优化 prefill：在同长度或近似长度组内使用 batch prefill，并结合 attention mask，减少 Python 循环。
4. 减少 `_repeat_kv()` 的物理扩展成本：可尝试使用 broadcast/view 方式或更适配 GQA 的 attention 写法。
5. 增强 cache 管理：当前是连续大 buffer，可进一步实现 block/page 级 KV 管理，降低长上下文与长 decode 的内存碎片风险。
6. 强化 CUDA 专用路径：针对 GPU 环境进一步验证 float16/bfloat16、SDPA/FlashAttention 后端和张量布局，减少不必要的数据搬运。
7. 增强 serving 策略：考虑 priority、deadline、prompt length bucket、max_new_tokens、stream_size 等字段，优化吞吐与尾延迟。

## 10. 总结

当前 `StudentEngine` 已完成 strict inference-engine track 所要求的核心能力：不依赖完整模型 API，手写 Qwen2 forward 与 greedy decode，并实现 KV cache、批量 decode、长度感知 batching 和 serving 接口。最终 GPU 完整评测得分为 83.44 / 100，其中 long context correctness 为 28.50 / 30，TTFT 为 17.41 / 20，runtime robustness 为 10.00 / 10，说明正确性、首 token 延迟和运行稳定性已经达到较好水平。

在 GPU、4 核、12G 的真实运行配置下，全流程峰值显存约 1348 MB，内存压力较小。主要性能瓶颈集中在 Python 手写 forward 的 kernel 调度开销、批量 decode 收益不足、prefill 长尾延迟和 serving p95 latency。后续若继续围绕 GPU attention kernel、prefix reuse、continuous batching 和 12G 内存预算内的 KV cache 管理优化，仍有明显提升空间。
