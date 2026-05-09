# OKX 多空信号扫描器

实时监控 OKX USDT 现货与永续合约，基于 15 分钟 K 线技术指标（成交量、RSI、MACD、布林带、KDJ）和 CoinGecko 叙事热度，筛选做多/做空机会，并通过飞书推送信号及 DeepSeek AI 分析。

## 功能特性

- 🔍 扫描所有 USDT 交易对（现货+合约）
- 📊 多重技术指标综合评分（最高9分）
- 🎭 叙事热度加权（CoinGecko Trending）
- 🤖 DeepSeek AI 智能分析（可选）
- 💰 动态止盈止损位
- 📈 信号历史验证（1h/4h/24h）
- 📱 飞书机器人实时推送
- 🗄️ SQLite 持久化存储
- 🐳 支持 Docker / Railway 部署

## 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MIN_VOLUME_USD` | 最小24h成交额(USD) | 2000000 |
| `VOL_MULTIPLIER` | 放量倍数阈值 | 1.5 |
| `ENABLE_SHORT` | 是否启用做空 | True |
| `LONG_SCORE_THRESHOLD` | 做多最低得分 | 5 |
| `SHORT_SCORE_THRESHOLD` | 做空最低得分 | 5 |
| `FEISHU_WEBHOOK` | 飞书机器人 Webhook URL | (必填) |
| `NARRATIVE_ENABLED` | 启用叙事热度 | True |
| `NARRATIVE_WEIGHT` | 叙事权重(1.0~2.5) | 1.5 |
| `ENABLE_DEEPSEEK` | 启用 DeepSeek 分析 | False |
| `DEEPSEEK_API_KEY` | DeepSeek API Key | (可选) |
| `DEEPSEEK_MODEL` | DeepSeek 模型 | deepseek-chat |
| `DB_PATH` | 数据库路径 | /app/data/signals.db |

完整配置请查看 `main.py` 中的 `Config` 类。

## 部署到 Railway

1. **Fork 本仓库** 到你的 GitHub。
2. 登录 [Railway](https://railway.app/) → **New Project** → **Deploy from GitHub repo**。
3. 选择你的仓库。
4. 添加必需的环境变量：`FEISHU_WEBHOOK`。
5. （可选）添加 `ENABLE_DEEPSEEK=True` 和 `DEEPSEEK_API_KEY`。
6. 点击 **Deploy**，Railway 会自动构建并运行。

Railway 会自动识别 `Dockerfile` 并启动容器。

## 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量（或写入 .env）
export FEISHU_WEBHOOK="https://open.feishu.cn/..."

# 运行
python main.py