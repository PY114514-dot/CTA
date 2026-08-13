# CTA 净值研究与 FOF 工作台

面向个人研究使用的私募 CTA 分析工具。项目把净值、曲线图片、PDF/Office 周报和机构因子材料整理为可复核的数据，再提供业绩分析、CTA 归因、产品排名、FOF 候选研究与投委会记录。

> 本项目输出用于研究辅助，不构成投资建议。图片识别、OCR/VLM 和模型推断结果必须经过人工复核，不能视为真实持仓或已验证的因子暴露。

## 主要能力

- **净值导入与业绩计算**：支持 CSV/XLSX、文本和曲线图片，计算累计收益、年化收益、年化波动、夏普、最大回撤与卡玛比率。
- **图表与材料提取**：支持 PNG、JPG、WebP、PDF、DOCX 和 PPTX；可裁切图表、取色、校准坐标并复核候选曲线。
- **CTA 分阶段归因**：基于已审核净值执行静态特征、动态 Beta、市场状态和非线性增量归因，并可冻结审计快照。
- **因子研究**：提供自建 CTA 因子库、外部机构因子材料、回归诊断、风险覆盖、板块贡献和导出能力。
- **产品知识库**：保存原始材料、产品身份、净值候选、结构化事实、审核状态与来源定位，确保研究结论可追溯。
- **FOF 工作台**：支持候选产品库、材料研究、可审计 Agent 对话、CTA 排名、组合建议、投委会审批和推荐后追踪。
- **研究报告**：生成 Markdown、PDF 和 XLSX 结果，保留方法、数据边界、警告和证据来源。

### 识别质量门

- 曲线识别使用全图宽度的连续路径搜索，并记录覆盖率、最大跳变和连续性指标；短图例或断裂片段不会自动变成净值。
- 坐标校准只接受人工锚点、带像素位置的 OCR 框或实际网格线。VLM 读到的刻度文字、`y_range` 和均匀间距不能单独生成像素锚点。
- 任何曲线、纵轴或横轴证据不足时，结果只作为候选，必须对照原图人工确认后才能进入研究。
- 多产品材料先完成产品/曲线绑定；文件名或单个 OCR 标签不会自动决定产品归属。

## 技术栈

- 后端：Python、FastAPI、SQLAlchemy、pandas、NumPy、SciPy、statsmodels、scikit-learn、OpenCV
- 前端：React、TypeScript、Vite、Ant Design、ECharts、KaTeX
- 本地存储：SQLite 与文件系统；运行数据默认不提交到 Git

## 目录结构

```text
backend/           FastAPI API、数据库模型、分析服务与测试
frontend/          React 研究界面
scripts/           批量导入与工程验证脚本
start.py           Windows 本地启动器
启动.bat           双击启动入口
```

## 本地安装

需要 Python 3.11+、Node.js 20+。以下命令以 PowerShell 为例。

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

cd ..\frontend
npm install
```

需要运行测试时，再安装开发依赖：

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

## 启动

推荐在项目根目录运行统一启动器：

```powershell
python start.py
```

Windows 也可以双击 `启动.bat`。启动后访问：

- 前端：<http://127.0.0.1:5275>
- API 文档：<http://127.0.0.1:8103/docs>
- 健康检查：<http://127.0.0.1:8103/api/health>

如需分别启动：

```powershell
# 终端 1
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8103

# 终端 2
cd frontend
npm run dev -- --host 127.0.0.1 --port 5275 --strictPort
```

端口被占用时，启动会直接报错；请关闭旧服务后重试。

## 可选模型配置

DeepSeek/兼容 OpenAI API 的地址、模型与令牌可以在应用设置中配置。PaddleOCR-VL 令牌只应放在启动后端的环境变量中：

```powershell
$env:PADDLEOCR_API_TOKEN = "你的令牌"
```

不要把令牌写入 README、前端配置或提交到 GitHub。

## 验证

运行项目验证入口：

```powershell
.\scripts\verify.ps1
```

也可以分别运行：

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest

cd ..\frontend
npm run build
```

## 数据与安全边界

- `data/`、`.env`、虚拟环境、前端依赖和构建产物已被 Git 忽略。
- 原始私募材料可能包含敏感信息；上传 GitHub 前应确认仓库可见性和待提交文件。
- 模型输出必须保留来源、置信度与人工审核状态，不能替代正式尽调。
