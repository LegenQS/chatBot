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
前往https://www.python.org/downloads/windows/下载，并勾选`Add Python to PATH`；安装完成后检查python版本号

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

## 3. 运行程序
```bash
streamlit run app.py
```

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
