# AShareAgents：面向 A 股市场的多智能体 LLM 交易框架

> 基于 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) 修改，将原始项目中的英文数据源替换为中文 A 股市场数据源，使其适用于沪深京三地上市公司的分析与交易决策。

## 与原项目的区别

TradingAgents 是一个优秀的多智能体 LLM 交易框架，但其数据源面向美股市场。**AShareAgents** 在此基础上做了以下修改：

### 情绪分析数据源

| 原项目 | AShareAgents |
| ------ | ------------ |
| Yahoo Finance 新闻 | **新浪财经**（Sina Finance）个股新闻 |
| Reddit（r/wallstreetbets、r/stocks 等） | **东方财富股吧**（East Money Guba）散户论坛 |
| StockTwits | 同上，合并至东方财富股吧 |

### 市场数据源

| 原项目 | AShareAgents |
|--------|-------------|
| Yahoo Finance K 线数据 | **东方财富** K 线 API（自动检测 A 股后优先使用） |
| Yahoo Finance 基本面 | **东方财富** datacenter API |
| Yahoo Finance 财报 | 东方财富（暂未实现，自动回退至 Yahoo Finance） |

### 全局新闻搜索

新增中文市场关键词：央行货币政策、A 股沪深 300、北向资金、人民币汇率等。

### 处理逻辑

- 当股票代码以 `.SZ`、`.SS` 或 `.BJ` 结尾时，**自动切换到中文数据源**
- `.SZ` 为深交所，`.SS` 为上交所，`.BJ` 为北交所
- 美股及其他市场代码保持原有逻辑不变

## 安装与使用

```bash
# 克隆仓库
git clone https://github.com/philzhxu/AshareAgents.git
cd AShareAgents

# 安装依赖
pip install .

# 安装额外依赖（中文数据源需要 TLS 指纹伪装）
pip install curl_cffi

# 运行
python main.py
```

然后输入 A 股代码（如 `000858.SZ` 五粮液、`601318.SS` 中国平安）即可。

## 原项目致谢

本项目的全部基础代码来自 [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)，感谢原作者的出色工作。

## 引用

如果本项目对您有帮助，请引用原论文：

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```

## 免责声明

本框架仅供研究用途。交易表现受多种因素影响，包括底层模型选择、温度参数、交易时段、数据质量及其他非确定性因素。**不构成任何金融、投资或交易建议。**
