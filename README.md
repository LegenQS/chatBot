# 🚀 项目安装指南（Windows）

---

# 🧩 第一步：安装 Python

## 1. 检查是否已安装

按 `Win + R` → 输入 `cmd` → 回车  
然后输入：

```bash
python --version
```

如果显示版本号（例如 Python 3.11.6），跳过下一步。

## 2. 如果尚未安装python
前往https://www.python.org/downloads/windows/ 下载，并勾选`Add Python to PATH`；安装完成后检查python版本号

# ⚙️ 第二步：运行Python
## 1. 创建python虚拟环境
命令行输入
```bash
python -m venv venv
```
创建成功后命令行中应出现(venv) 前缀

## 2. 安装项目依赖
在项目目录下运行
```bash
pip install -r requirements.txt
```
如果没有添加到PATH导致上述命令报错，你可以尝试
```bash
python -m pip install -r requirements.txt
```

### Troubleshoot
如果报错提示C/C++ not found，可以通过以下两种方式：
- 如果有安装conda，可以使用conda代替venv
  ```bash
  conda create -n chatbot python=3.10
  conda activate chatbot
  conda install faiss-cpu
  pip install -r requirements.txt  # rest of packages
  ```
- 如果没有，前往https://visualstudio.microsoft.com/visual-cpp-build-tools/ 下载VS C组件
  - 下载“Build Tools for Visual Studio”.
  - 安装时提示勾选
    - （必须）Desktop development with C++
    - （可选）MSVC v143 – latest C++ compiler and Windows 10/11 SDK
  - 安装完成后在命令行输入`cl`,应当出现
    ```bash
    Microsoft (R) C/C++ Optimizing Compiler Version 19.36...
    ```

## 3. 运行程序
```bash
streamlit run app.py
```
> 首次运行时会自动完成两件事（均为本地运行，无需任何 API Key）：
> 1. 从 Hugging Face 自动下载本地大模型（仅下载一次，之后离线可用）；
> 2. 自动构建检索索引（`manual.index`）。
>
> 请耐心等待首次加载完成；之后再次启动会很快。
>
> **停止程序**：点击左侧边栏的「🛑 停止并退出」按钮即可关闭服务器并退出，无需在终端按 Ctrl+C。
> （模型文件会一直保存在 `model/` 目录，不会每次重新下载。）

### 模型档位（速度 vs. 质量）
可在**左侧边栏的“模型”下拉框**中随时切换档位，切换后会自动下载（仅一次）并加载：

| 档位 | 模型 | 大小 | 适用场景 |
| --- | --- | --- | --- |
| `fast` ⚡ | Qwen2.5-**3B**-Instruct-Q4 | ~2GB | 仅 CPU，追求速度 |
| `quality` ⭐ | Qwen2.5-**7B**-Instruct-Q4 | ~4.7GB | 有 GPU，质量更好 |
| `powerful` 🚀 | Qwen2.5-**14B**-Instruct-Q4 | ~9GB | 显存 10GB+，质量最佳 |

首次进入时的**默认档位**会根据硬件自动判断（检测到 GPU → `quality`，否则 → `fast`），也可用环境变量 `MODEL_TIER` 指定默认值：
```bash
# Windows PowerShell
$env:MODEL_TIER="powerful"; streamlit run app.py
# macOS / Linux
MODEL_TIER=powerful streamlit run app.py
```
当前档位与加速器会显示在左侧边栏（例如 `⚙️ powerful · GPU`）。
> 提示：14B 在纯 CPU 上会很慢，建议搭配下方的 GPU 加速使用。切换档位时同一时间只会占用一个模型的内存/显存。

### ⚡ 启用 NVIDIA GPU 加速（Windows，强烈推荐）
默认 `pip install llama-cpp-python` 安装的是 **仅 CPU** 版本；要让 7B 模型跑得又快又好，需要安装 **CUDA 版本**（请把 `cu124` 换成与你显卡驱动匹配的 CUDA 版本，如 `cu121`/`cu122`/`cu123`/`cu124`）：
```bash
pip install --upgrade --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```
安装后程序会自动把模型层卸载到 GPU（`n_gpu_layers=-1`）。若未安装 CUDA 版本，7B 会退回到 CPU 运行，速度较慢——此时可改用 `MODEL_TIER=fast`。
> Apple Silicon（Mac）无需额外操作，会自动使用 Metal GPU。

# 📃 附录：文档更新
如果需要更新源文档，即根目录下`original/error-diagnose-and-maintenance-instruction.docx`或`original/heating-system-checking-instruction.docx`
或添加新的源文件，应遵循以下步骤

## 1. 文件格式
文档标题请不要使用自动分段格式以免识别失效，应当使用的格式如下，其中大标题和小标题顶格无空格，次标题应当空4格，剩余正文中分点可适用1. 或者1 但
请不要使用1、顿号作为分隔符。

一、大标题
1、小标题
    1、次标题

## 2. 替换文件路径
在`parse_doc.py`中将`INPUT_DOCX`和`OUTPUT_JSON`路径替换为修改文件路径，并运行
```bash
python parse_doc.py
```

## 3. 更新index
- 如果新添加了文档，请在`build_index.py`中更新`JSON_PATHS`参数，添加所有需要的文件
- 如果只是更新了已存在文档，可直接运行
```bash
python build_index.py
```
