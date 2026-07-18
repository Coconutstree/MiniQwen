# 目标
实现一个最小可运行的 Pytorch 推理流程，文件入口：whiteboard_llm_inference.py

# 具体实现

## Tokenier
你需要实现从文本变为 token_id 的过程，接口名为 simple_tokenizer()

## Embedding
你需要实现从 token_id 变为 hidden 向量的过程，接口名为 embedding_lookup()

## 归一化
实现 RMSNorm 归一化方法，接口名为 rms_norm()

## Attention 注意力机制
### Q/K/V 计算
通过矩阵乘法计算来实现

### 位置编码
实现 RoPE 位置编码方法，接口名为 apply_rope()

### 因果注意力
实现因果注意力，接口名为 scaled_dot_product_attention()

### Attenion 后残差连接
在主函数 main() 中实现残差的张量相加

## FFN
### FFN 计算
实现FFN计算，接口名为 toy_ffn()
再进行残差连接，在主函数 main()中实现

## 输出预测
输出预测，接口名为 lm_head()

# main()函数流程
1. Tokenizer
把文本转换为 token_id
2. Embedding
把token_id 转换为 hidden states
3. RMSNorm
对每个token的hidden向量做归一化
4. Q/K/V Projection
对于归一化后的 hidden states计算Q，K，V
5. RopE
给Q/K加入位置信息
6. Casual Attention
计算 $QK^T/ \sqrt{d_k}$ 加入Casual Mask,再经过softmax并对V进行加权求和
7. Attention Residual
$x_1 = x+Attention( RMSNorm(x))$
8. RMSNorm + FFN +Residual:
$x_2 = x_1+FFN(RMSNorm(x_1))$
9. LLM Head
用最后一个token的hidden state 预测下一个token

# 注意
添加打印形状日志
