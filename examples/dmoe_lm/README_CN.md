# dMoE 语言模型示例：昇腾 NPU

本示例在单张昇腾 NPU 上端到端训练并运行一个小型 GPT 风格 Transformer。这里的 Transformer 是一种以注意力机制为核心的神经网络结构，常用于语言模型。模型的前馈网络块采用 MegaBlocks 的 dMoE 层。dMoE 是不丢弃任何 token 的混合专家层，其中 token 指文本被切分后的最小单位。本示例实际跑通的是 MegaBlocks 已支持的 NPU 链路，具体包括三部分：专家计算走分组矩阵乘法并落到 `torch_npu.npu_grouped_matmul` 上，token 按专家重排走 triton-ascend 内核，路由的负载均衡由辅助损失约束。参数精度为 bf16 或 fp16。

模型刻意保持小规模，采用可学习词嵌入、手写因果注意力机制以及 dMoE 块，目的是让脚本易于阅读，且无需引入完整的训练框架。因果注意力指每个位置只能看到自身及之前位置的信息，符合语言模型逐词生成的设定。默认配置约 2.09 亿参数，词嵌入与输出层不共享权重，具体为 6 层、隐藏维度 512、8 个专家且每个 token 选取得分最高的 2 个专家、MLP 采用 GLU 结构、序列长度 512。这里的 MLP 指前馈网络，GLU 指带门控的线性单元结构。

## 环境要求

- CANN 工具包：先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 完成环境加载，并准备一张 910B 级别的 NPU。
- `torch >= 2.7.0` 以及与之匹配的 `torch_npu`，另需 `triton-ascend`。
- 从本仓库安装 MegaBlocks：`pip install -e .`
- 数据依赖：执行 `pip install datasets transformers` 安装。仅当使用 `--dataset-source modelscope` 路径时，才需要额外安装 `modelscope` 和 `pyarrow`。

## 1. 准备数据

```bash
# 默认路径：Hugging Face 上的 wikitext-2-raw-v1 数据集加 GPT-2 分词器。
export HF_ENDPOINT=https://hf-mirror.com  # 无法直连 huggingface.co 时改走镜像站
python data.py prepare --data-dir data --dataset-source hf

# 备选路径：通过 ModelScope SDK 加载文本。
python data.py prepare --data-dir data --dataset-source modelscope
```

上述命令会写出扁平的 uint16 token 流文件 `data/tokens_train.bin` 和 `data/tokens_val.bin`，并把分词器保存在 `data/tokenizer/` 目录下。uint16 指每个 token 用 16 位无符号整数存储。任何其他 uint16 token 文件都可以直接使用，其中训练数据通过 `--data-path` 指定，验证数据通过 `--eval-data-path` 指定。

## 2. 训练

```bash
python train.py --steps 1000 --save checkpoints
```

当 token 文件缺失时，`train.py` 会自动先准备数据集。常用参数包括 `--seq-length`、`--micro-batch-size`、`--num-layers`、`--num-experts`、`--top-k`、`--mlp-type {mlp,glu}`、`--dtype {bf16,fp16}`、`--lr`、`--warmup-steps`、`--load` 和 `--eval-interval`，其中 `--load` 用于从已有检查点恢复训练，`--eval-interval` 用于控制验证间隔。完整参数列表见 `--help` 输出。

训练进度以 `step=... lm_loss=... aux_loss=... lr=... tokens_per_sec=...` 的形式逐行记录，并周期性输出 `val_loss=...` 行。各字段含义依次为训练步数、语言模型损失、负载均衡辅助损失、当前学习率和每秒处理的 token 数。启动时输出的第一行形如 `device: npu:0 (torch_npu ...)`，会标明实际使用的设备及其 torch_npu 版本。

不下载数据集、用随机 token 快速冒烟运行的方法如下：

```bash
python -c "import numpy as np; np.random.default_rng(0).integers(0,50257,size=200000,dtype=np.uint16).tofile('tokens.bin')"
python train.py --data-path tokens.bin --steps 20 --num-layers 2 --log-interval 5
```

## 3. 生成文本

```bash
python generate.py --checkpoint checkpoints/checkpoint.pt --prompt 'Once upon a time'
```

采样相关参数包括 `--max-new-tokens`、`--temperature`、`--top-k`、`--top-p` 和 `--greedy`。其中 `--max-new-tokens` 控制最多生成多少个新 token，`--temperature` 控制采样随机性，`--top-k` 和 `--top-p` 控制每一步的候选范围，`--greedy` 表示每一步都直接取概率最高的 token。若不提供 `--prompt`，脚本会取验证集开头的 token 作为上下文。

## 故障排查

在部分 CANN 9.1.0 与 torch_npu 2.7.x 的安装组合上，首次启动 triton-ascend 内核时会在编译 triton 的 `npu_utils` 兼容层代码时失败，报错信息包含 `'aclOpExecutor' was not declared in this scope`。解决办法是改走一个包装脚本进行编译，该脚本会强制包含一份前向类型声明。请在运行任何脚本之前先完成以下设置：

```bash
mkdir -p .triton_cc
echo "typedef struct aclOpExecutor aclOpExecutor;" > .triton_cc/stub.h
printf '#!/bin/sh\nexec g++ -include %s/.triton_cc/stub.h "$@"\n' "$PWD" > .triton_cc/gxx.sh
chmod +x .triton_cc/gxx.sh
export CC="$PWD/.triton_cc/gxx.sh"
```

较新的 CANN 版本已经自带该声明，不需要这一步。

## 预期与限制

- 本示例的目标是演示已支持的 NPU dMoE 链路能够端到端跑通，也就是训练损失能够下降且生成能够产出文本，而不是训练出一个效果好的语言模型。原因是 wikitext-2 数据量很小，默认训练步数也较少。
- 标准 MoE 采用基于容量的做法，专家计算的每组容量有上限，超出的 token 会被丢弃。这类标准 MoE、`mlp_impl='sparse'` 的稀疏实现以及多卡专家并行目前在 NPU 上均不受支持。训练脚本检测到 `torchrun` 拉起的多进程环境时会直接拒绝运行。
- 必须使用半精度，即 bf16 或 fp16。
- 第一个训练步会比较慢，因为 triton-ascend 需要先编译一次 token 重排内核，并把缓存保存在 `.triton_cache/` 目录下，后续运行可直接复用。
- 相同脚本也可以在 CUDA GPU 上运行，此时需要安装 `megablocks[gg]` 以提供分组矩阵乘法依赖，但本文档验证的不是 CUDA 路径。
