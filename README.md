# MicroGPT 语言模型训练与推理压测

## 📖 项目简介
本项目基于 Karpathy 的 `microgpt` 思想，使用 PyTorch 从零构建了一个字符级（Char-level）GPT 模型。项目采用**模块化解耦**设计，包含完整的数据预处理、多头注意力机制（Multi-Head Attention）训练、验证集误差评估以及推理吞吐量压测流程。支持在 NVIDIA GPU (CUDA) 上进行高效训练与推理。

## 💻 环境依赖
- **操作系统**: WSL2 (Ubuntu)
- **硬件**: NVIDIA RTX 5070 Laptop GPU (8GB)
- **驱动/CUDA**: CUDA 12.8 (Blackwell 架构)
- **核心库**: PyTorch 2.11.0+cu128, Python 3.11

## 📂 代码结构
为了让项目具备更好的可维护性和可扩展性，我对代码进行了解耦：
- `data.py`: 负责读取数据、构建词表、划分训练/验证集，以及生成 Batch (自动补全/截断)。
- `model.py`: 定义 Multi-Head Attention 及完整的 MicroGPT 网络结构。
- `train.py`: 主程序入口，包含模型初始化、训练循环、验证集评估、以及吞吐量（tokens/s）压测。
- `gen_data.py`: 辅助脚本，用于在无网络环境下生成合成数据（Synthetic Data）进行早期连通性测试。
- `microgpt.py`: Karpathy 提供的纯 Python 原版参考代码。

---

## 🚧 遇到的难点与工程化解决方式

### 难点 1：外部网络壁垒与工程鲁棒性（Robustness）
- **问题**：Karpathy 原版代码中使用了 `urllib.request` 从 GitHub 自动拉取 `names.txt` 数据集。但在国内网络环境下，访问 `raw.githubusercontent.com` 极易发生 `ConnectionResetError`（连接重置），导致程序阻塞。
- **解决**：为了保证项目的离线可运行性，采用了**“本地兜底（Fallback）”**方案：直接通过浏览器下载完整数据集并作为本地依赖 `input.txt` 提供。这让我深刻认识到在工程落地时，不能完全依赖外部网络环境，必须要有本地数据隔离方案。

### 难点 2：极小样本导致的严重过拟合
- **问题**：在最初未接入完整数据集时，仅使用 30 条数据测试，导致模型严重**过拟合**（Train Loss 降至 0.05，但验证集 Loss 飙升至 7.77，困惑度高达 2380，失去泛化能力）。
- **解决**：编写 `gen_data.py` 生成 500 条合成数据（Synthetic Data）作为 Mock 测试，成功将验证集 Loss 拉回 3.17，证明了代码链路的正确性。后续顺利切换至 3.2 万条真实数据集，彻底解决了过拟合。

### 难点 3：自回归生成的重复循环（Repetition Loop）
- **问题**：在使用真实数据集后，由于数据集去除了换行符（无 EOS 结束符），模型生成出 `nnnnnn` 或 `amamama` 这种死循环乱码，陷入局部最优解。
- **解决**：在 `generate` 推理函数中引入了 **温度缩放（Temperature Scaling）** 和 **重复惩罚（Repetition Penalty）** 机制，成功打破了死循环，使模型生成了具有多样性的词汇。

### 难点 4：纯 Python 惩罚导致 GPU 并行失效（核心性能瓶颈）
- **问题**：最初使用 `for` 循环遍历历史 Token 进行惩罚修改，导致 Host 与 Device 频繁通信，破坏了 GPU 的并行计算，**吞吐量从 290 暴跌至 50 tokens/s**。
- **解决**：使用 PyTorch 的 **张量向量化（Tensor Vectorization）** 技术，利用 `torch.unique` 提取重复 Token 索引，结合 `torch.where` 算子，完全在 GPU 显存中并行完成惩罚逻辑，**成功将吞吐量恢复到 219+ tokens/s**。

---

## 📊 最终验收指标
基于 **32,000 条**真实英文名字数据集，训练 10000 步（参数配置：`n_embd=64, n_head=4, block_size=16`）：

- **最终验证集 Loss**: **1.4884**
- **验证集困惑度 (Perplexity)**: **4.39**
- **推理生成耗时**: 生成 100 个 token 约 0.4556 秒
- **推理吞吐量 (Throughput)**: **219.48 tokens/s**
- **生成示例**: `xayley`, `xllely`, `qylyea`, `ygyzll` 等。

---

**关于 vLLM 的思考**：虽然因本地环境（RTX 5070 Blackwell）未直接部署 vLLM 框架，但通过手写推理压测，我深刻体会到了 **内存访问效率** 和 **算子并行化** 对吞吐量的决定性影响。未来若引入 vLLM 的 PagedAttention 技术处理 KV Cache，吞吐量将有质的飞跃。

## 🚀 如何复现
```bash
# 激活环境
conda activate microgpt
# 运行主程序（包含训练、评估、压测全流程）
python train.py