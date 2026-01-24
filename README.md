# ⚖️ DEBIT_SYNC // 智能账单对账工具

**DEBIT_SYNC** 是一款采用“复古未来极简主义 (Retro-Future Minimalism)”设计风格的自动化对账工具。它专为解决中国工商银行 (ICBC) 流水与微信支付明细之间的对账痛点而设计，通过精密算法实现当日流水的全自动配对。

![Aesthetics](https://img.shields.io/badge/Aesthetics-Retro--Future-orange)
![Python](https://img.shields.io/badge/Python-3.14-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

## 🚀 核心功能

### 1. 智能账单对账 (DEBIT_SYNC)
- **双 Excel 自动对账**：支持工行 XLSX 流水与微信导出明细的直接对位，无需 PDF 繁琐步骤。
- **Multiset 匹配算法**：按日期对每一笔交易进行排序对比，精准识别漏项。
- **多维度明细追踪**：在对账异常时，自动展示当日双端的“对方户名”、“交易内容”与“商品详情”。
- **人机协同审核**：支持一键审核已确认的异常流水，状态实时同步，主表动态变绿。
- **复古未来主义视觉**：黑曜石底色配合琥珀色荧光点缀，带给您极简且硬核的对账体验。

## 🛠️ 安装与运行

1. **环境准备**：
   项目采用 `uv` 进行依赖管理。
   ```bash
   pip install uv
   ```

2. **安装依赖**：
   ```bash
   uv sync
   ```

3. **启动对账工具**：
   ```bash
   uv run streamlit run app.py
   ```

## 📂 项目结构 (File Structure)

本仓库保持极致极简，仅包含核心对账逻辑相关文件：

- **`app.py`**：核心应用程序。包含了复古未来感 UI 的定义、双 Excel 解析内核、以及按日自动配对的对账算法。
- **`reconcile_bills.py`**：辅助对账逻辑库。
- **`pyproject.toml`**：项目依赖配置文件。
- **`.gitignore`**：隐私防护罩。配置了严格的过滤规则，防止任何用户信息和临时缓存进入版本库。
- **`README.md`**：您当前正在阅读的说明文档。

## 🔒 隐私安全

- **本地解析**：所有账单数据仅在内存中处理，绝不上传至任何云端服务器。
- **.gitignore 保护**：默认配置严格过滤所有 `.xlsx` 和 `.pdf` 文件，确保您的隐私数据不会进入 Git 版本库。

## 🎨 视觉风格说明

本项目遵循 **现代极简 + 复古未来** 的基调：
- **字体**：数据部分采用 `JetBrains Mono`，确保对账时的机械精准感。
- **布局**：一像素边框、直角逻辑、零圆角阴影。
- **色彩**：黑曜石 (#050505) 为底，琥珀色 (#FFAA00) 激活。

---
*Created with Passion for Precise Finance.*
