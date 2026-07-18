# 目标
在现有的模型上新建一个文件，实现Qwen2.5-0.5B-Instruct模型
1. 加载本地Qwen模型文件
2. 去出tokenier,config和真实权重
3. 按照Qwen结构手写一次完整forward
4. 与Transfomers 标准forward比较结果

和whiteboard_llm_inference.py的变化在于whiteboard_llm_inference.py使用手写参数；真实模型版本使用已经训练好的Qwen参数:Qwen2.5-0.5B-Instruct，但是forward过程中的主要计算过程仍然需要自己实现
# 具体实现

## 加载Qwen
使用`model = AutoModelForCausalLM.from_pretrained(model_path)`

## 本地模型包与Transformers 的分工
本地模型目录主要提供：
1. tokenizer文件：文本与token id 的转换规则
2. config.json ：模型层数，hidden size,head数等配置
3. model.safetensors: 训练好的真实模型权重
4. generation config 等其他配置文件

Transformers库主要提供：
1. Qwen2ForCausalLM 等模型结构代码
2. Attention,MLP ,RMSNorm等模块的标准实现
3. forward()和生成流程相关接口
4. 模型文件读取和权重加载功能

## 配置
- HF_MODEL_NAME：模型名称，也可以设置为解压后的本地模型目录
- HF_DEVICE：使用 cpu 或 cuda
- CUDA_VISIBLE_DEVICES：指定程序可以看到的 GPU
- HF_LOCAL_FILES_ONLY：设置为1时只读取本地文件，不尝试联网
- HF_HOME：Hugging Face 本地缓存根目录

**注意**
1. RoPE中旋转方式是前半维与后半维组合旋转，与现有实现不同，需要修改

2. 需要手写forward 与 Transformers 标准forward对齐验证。输出shape需要完全一致，比较中间层和最终logits的数值误差。

## 目标产出
在whiteboard_llm_inference.py的基础上，加载Qwen2.5-0.5B-Instruct 的tokenizer,config和真实权重，手写完成一次input_ids到logits，并且与Transformers标准forward进行对齐验证

实现时重点检查
- inputs_id、hidden_states、logits 的 shape
- Linear 权重方向是否正确：$[out_features,in_features]$
- Q/K/V projection 后是否正确 reshape 成多头形式
- Qwen 中 K/V head 数量少于 Q head 数量，是否正确 repeat_kv
- manual forward 和 official forward 的 logits 是否对齐
