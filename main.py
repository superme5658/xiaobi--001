import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from functools import wraps
from typing import Optional, Dict, List, Tuple
import schedule
import sqlite3
from contextlib import contextmanager

# ============================================================
# 配置类
# ============================================================
class Config:
    # API配置
    OKX_BASE_URL = "https://www.okx.com"
    REQUEST_TIMEOUT = 10
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    CONCURRENT_WORKERS = 10
    
    # 扫描配置
    SCAN_INTERVAL = 15
    KLINE_LIMIT = 100
    
    # 通用筛选参数
    MIN_VOLUME_USD = int(os.getenv("MIN_VOLUME_USD", "2000000"))
    VOL_MULTIPLIER = float(os.getenv("VOL_MULTIPLIER", "1.5"))
    MAX_VOLUME_RATIO = float(os.getenv("MAX_VOLUME_RATIO", "20"))
    
    # ========== 做多信号配置 ==========
    LONG_SCORE_THRESHOLD = int(os.getenv("LONG_SCORE_THRESHOLD", "5"))
    LONG_MIN_VOLUME_RATIO = float(os.getenv("LONG_MIN_VOLUME_RATIO", "1.2"))
    LONG_MIN_CHANGE_15M = float(os.getenv("LONG_MIN_CHANGE_15M", "0.5"))
    LONG_MAX_CHANGE_15M = float(os.getenv("LONG_MAX_CHANGE_15M", "12.0"))
    LONG_MIN_RSI = float(os.getenv("LONG_MIN_RSI", "35"))
    LONG_MAX_RSI = float(os.getenv("LONG_MAX_RSI", "75"))
    
    # 做多止盈止损
    LONG_STOP_LOSS_PERCENT = float(os.getenv("LONG_STOP_LOSS_PERCENT", "3"))
    LONG_TAKE_PROFIT_1 = float(os.getenv("LONG_TAKE_PROFIT_1", "5"))
    LONG_TAKE_PROFIT_2 = float(os.getenv("LONG_TAKE_PROFIT_2", "10"))
    LONG_TAKE_PROFIT_3 = float(os.getenv("LONG_TAKE_PROFIT_3", "20"))
    
    # ========== 做空信号配置 ==========
    ENABLE_SHORT = os.getenv("ENABLE_SHORT", "True").lower() == "true"
    SHORT_SCORE_THRESHOLD = int(os.getenv("SHORT_SCORE_THRESHOLD", "5"))
    SHORT_MIN_VOLUME_RATIO = float(os.getenv("SHORT_MIN_VOLUME_RATIO", "1.2"))
    SHORT_MIN_CHANGE_15M = float(os.getenv("SHORT_MIN_CHANGE_15M", "-2.0"))
    SHORT_MAX_CHANGE_15M = float(os.getenv("SHORT_MAX_CHANGE_15M", "-0.5"))
    SHORT_MIN_RSI = float(os.getenv("SHORT_MIN_RSI", "65"))
    SHORT_MAX_RSI = float(os.getenv("SHORT_MAX_RSI", "85"))
    
    # 做空止盈止损
    SHORT_STOP_LOSS_PERCENT = float(os.getenv("SHORT_STOP_LOSS_PERCENT", "3"))
    SHORT_TAKE_PROFIT_1 = float(os.getenv("SHORT_TAKE_PROFIT_1", "5"))
    SHORT_TAKE_PROFIT_2 = float(os.getenv("SHORT_TAKE_PROFIT_2", "10"))
    SHORT_TAKE_PROFIT_3 = float(os.getenv("SHORT_TAKE_PROFIT_3", "20"))
    
    # 动态调整
    DYNAMIC_TP = os.getenv("DYNAMIC_TP", "True").lower() == "true"
    
    # 技术指标参数
    BB_PERIOD = 20
    BB_STD = 2
    KDJ_PERIOD = 9
    RSI_PERIOD = 14
    
    # 推送冷却
    SIGNAL_COOLDOWN = 60 * 60
    SCORE_UPGRADE_THRESHOLD = 2
    
    # 市场监控配置
    MARKET_CHECK_INTERVAL = 60
    LOW_VOLUME_THRESHOLD = 500
    HIGH_VOLUME_THRESHOLD = 2000
    
    # 信号验证配置
    VERIFY_INTERVALS = [1, 4, 24]
    VERIFY_CHECK_INTERVAL = 5
    
    # 数据库配置
    DB_PATH = os.getenv("DB_PATH", "/app/data/signals.db")
    
    # 飞书
    FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK", "")
    
    # 叙事热度配置 (CoinGecko Trending)
    NARRATIVE_ENABLED = os.getenv("NARRATIVE_ENABLED", "True").lower() == "true"
    NARRATIVE_WEIGHT = float(os.getenv("NARRATIVE_WEIGHT", "1.5"))
    
    # ========== DeepSeek 分析配置 ==========
    ENABLE_DEEPSEEK = os.getenv("ENABLE_DEEPSEEK", "False").lower() == "true"
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    DEEPSEEK_TIMEOUT = int(os.getenv("DEEPSEEK_TIMEOUT", "15"))
    
    # 日志级别
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据库管理
# ============================================================
def adapt_datetime(dt: datetime) -> str:
    return dt.isoformat()

def convert_datetime(s: bytes) -> datetime:
    return datetime.fromisoformat(s.decode())

sqlite3.register_adapter(datetime, adapt_datetime)
sqlite3.register_converter("timestamp", convert_datetime)

@contextmanager
def get_db_connection():
    db_dir = os.path.dirname(Config.DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(Config.DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class Database:
    def __init__(self):
        self.init_db()
        self.fix_orphan_records()
    
    def fix_orphan_records(self):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM signal_verifications 
                WHERE signal_id NOT IN (SELECT id FROM signals)
            """)
            conn.commit()
    
    def init_db(self):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inst_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    signal_time TIMESTAMP NOT NULL,
                    price REAL NOT NULL,
                    score INTEGER NOT NULL,
                    narrative_multiplier REAL DEFAULT 1.0,
                    volume_ratio REAL,
                    change_15m REAL,
                    change_24h REAL,
                    rsi REAL,
                    bb_position REAL,
                    kdj_k REAL,
                    kdj_d REAL,
                    kdj_j REAL,
                    quality_reason TEXT,
                    stop_loss REAL,
                    take_profit_1 REAL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id INTEGER NOT NULL,
                    inst_id TEXT NOT NULL,
                    signal_time TIMESTAMP NOT NULL,
                    signal_price REAL NOT NULL,
                    direction TEXT NOT NULL,
                    verify_hours INTEGER NOT NULL,
                    verify_time TIMESTAMP,
                    verify_price REAL,
                    change_percent REAL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_time TIMESTAMP NOT NULL,
                    btc_volume REAL,
                    btc_change REAL,
                    active_signals_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(signal_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals(direction)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_verify_status ON signal_verifications(status)")
            conn.commit()
            logger.info("数据库初始化完成")
    
    def save_signal(self, signal_data: dict) -> Optional[int]:
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO signals (
                        inst_id, direction, signal_time, price, score, narrative_multiplier,
                        volume_ratio, change_15m, change_24h, rsi, bb_position,
                        kdj_k, kdj_d, kdj_j, quality_reason,
                        stop_loss, take_profit_1, take_profit_2, take_profit_3
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_data['inst_id'],
                    signal_data['direction'],
                    signal_data['signal_time'],
                    signal_data['price'],
                    signal_data['score'],
                    signal_data.get('narrative_multiplier', 1.0),
                    signal_data.get('volume_ratio'),
                    signal_data.get('change_15m'),
                    signal_data.get('change_24h'),
                    signal_data.get('rsi'),
                    signal_data.get('bb_position'),
                    signal_data.get('kdj_k'),
                    signal_data.get('kdj_d'),
                    signal_data.get('kdj_j'),
                    signal_data.get('quality_reason', ''),
                    signal_data.get('stop_loss'),
                    signal_data.get('take_profit_1'),
                    signal_data.get('take_profit_2'),
                    signal_data.get('take_profit_3')
                ))
                signal_id = cursor.lastrowid
                conn.commit()
                return signal_id
        except Exception as e:
            logger.error(f"保存信号失败: {e}")
            return None
    
    def add_verification_tasks(self, signal_id: int, inst_id: str, direction: str, 
                                signal_time: datetime, signal_price: float):
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                for hours in Config.VERIFY_INTERVALS:
                    verify_time = signal_time + timedelta(hours=hours)
                    cursor.execute("""
                        INSERT INTO signal_verifications (
                            signal_id, inst_id, signal_time, signal_price, direction,
                            verify_hours, verify_time, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                    """, (signal_id, inst_id, signal_time, signal_price, direction, hours, verify_time))
                conn.commit()
        except Exception as e:
            logger.error(f"添加验证任务失败: {e}")
    
    def get_stats(self) -> dict:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM signals")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM signals WHERE direction = 'LONG'")
            long_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM signals WHERE direction = 'SHORT'")
            short_count = cursor.fetchone()[0]
            return {'total': total, 'long': long_count, 'short': short_count}


db = Database()


# ============================================================
# OKX API
# ============================================================
def retry_on_failure(max_retries=Config.MAX_RETRIES, delay=Config.RETRY_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"{func.__name__} 失败: {e}")
            return None
        return wrapper
    return decorator


@retry_on_failure()
def _okx_request(url: str, params: dict = None) -> Optional[dict]:
    response = requests.get(url, params=params, timeout=Config.REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if data.get("code") != "0":
        raise ValueError(f"API错误: {data.get('msg')}")
    return data


def get_spot_symbols() -> List[str]:
    data = _okx_request(f"{Config.OKX_BASE_URL}/api/v5/public/instruments", 
                        params={"instType": "SPOT"})
    if not data:
        return []
    return [d["instId"] for d in data.get("data", []) 
            if d["instId"].endswith("-USDT") and d["state"] == "live"]


def get_swap_symbols() -> List[str]:
    data = _okx_request(f"{Config.OKX_BASE_URL}/api/v5/public/instruments",
                        params={"instType": "SWAP"})
    if not data:
        return []
    return [d["instId"] for d in data.get("data", [])
            if d["instId"].endswith("-USDT-SWAP") and d["state"] == "live"]


@retry_on_failure()
def get_ticker(inst_id: str) -> Optional[dict]:
    data = _okx_request(f"{Config.OKX_BASE_URL}/api/v5/market/ticker",
                        params={"instId": inst_id})
    if data and data.get("data"):
        return data["data"][0]
    return None


@retry_on_failure()
def get_klines(inst_id: str, bar: str = "15m", limit: int = None) -> Optional[pd.DataFrame]:
    if limit is None:
        limit = Config.KLINE_LIMIT
    
    data = _okx_request(f"{Config.OKX_BASE_URL}/api/v5/market/candles",
                        params={"instId": inst_id, "bar": bar, "limit": limit})
    
    if not data or not data.get("data"):
        return None
    
    df = pd.DataFrame(data["data"], columns=["ts", "o", "h", "l", "c", "vol", 
                                              "volCcy", "volCcyQuote", "confirm"])
    numeric_cols = ["ts", "o", "h", "l", "c", "vol", "volCcyQuote"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.sort_values("ts").reset_index(drop=True)
    
    if len(df) < 55:
        return None
    
    return df


@retry_on_failure()
def get_current_price(inst_id: str) -> Optional[float]:
    ticker = get_ticker(inst_id)
    if ticker:
        return float(ticker.get("last", 0))
    return None


# ============================================================
# 技术指标计算
# ============================================================
def calc_ema(series: pd.Series, period: int) -> pd.Series:
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if len(series) < period + 1:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    if len(series) < slow:
        return pd.Series([np.nan] * len(series)), pd.Series([np.nan] * len(series))
    
    ema_fast = calc_ema(series, fast)
    ema_slow = calc_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    return macd_line, signal_line


def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if len(series) < period:
        return pd.Series([np.nan] * len(series)), pd.Series([np.nan] * len(series)), pd.Series([np.nan] * len(series))
    
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def calc_kdj(df: pd.DataFrame, period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if len(df) < period:
        return pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df))
    
    low_min = df['l'].rolling(window=period).min()
    high_max = df['h'].rolling(window=period).max()
    rsv = (df['c'] - low_min) / (high_max - low_min + 1e-9) * 100
    
    k = pd.Series(50.0, index=df.index)
    d = pd.Series(50.0, index=df.index)
    
    for i in range(period, len(df)):
        k.iloc[i] = 2/3 * k.iloc[i-1] + 1/3 * rsv.iloc[i]
        d.iloc[i] = 2/3 * d.iloc[i-1] + 1/3 * k.iloc[i]
    
    j = 3 * k - 2 * d
    return k, d, j


# ============================================================
# 叙事热度模块 (CoinGecko Trending API)
# ============================================================
def get_top_narratives() -> Dict[str, float]:
    if not Config.NARRATIVE_ENABLED:
        return {}
    
    try:
        url = "https://api.coingecko.com/api/v3/search/trending"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=15)
        
        if resp.status_code != 200:
            logger.warning(f"Trending API 请求失败，状态码: {resp.status_code}")
            return {}
        
        data = resp.json()
        trending_coins = data.get("coins", [])
        
        if not trending_coins:
            logger.warning("Trending API 返回空数据")
            return {}
        
        narrative_keywords = {
            "ai": ["ai", "agent", "gpt", "llm", "neural", "intelligence", "skyai"],
            "meme": ["meme", "doge", "shib", "pepe", "wif", "floki"],
            "rwa": ["rwa", "realworld", "real world", "tokenized", "land", "property"],
            "gaming": ["game", "gaming", "metaverse", "play", "arena", "hero"],
            "defi": ["defi", "lending", "yield", "swap", "perpetual"],
            "layer2": ["layer2", "l2", "arbitrum", "optimism", "zksync", "polygon"],
            "solana": ["sol", "solana"],
            "mcp": ["mcp", "model context", "protocol"],
        }
        
        narrative_count = {n: 0 for n in narrative_keywords}
        
        for coin in trending_coins[:15]:
            coin_data = coin.get("item", {})
            coin_name = (coin_data.get("name", "") + " " + coin_data.get("symbol", "")).lower()
            
            for narrative, keywords in narrative_keywords.items():
                for kw in keywords:
                    if kw in coin_name:
                        narrative_count[narrative] += 1
                        break
        
        max_count = max(narrative_count.values()) if narrative_count else 1
        narrative_scores = {}
        for narrative, count in narrative_count.items():
            if count > 0:
                score = min(100, int(count * (100 / max_count)))
                narrative_scores[narrative] = score
        
        if narrative_scores:
            logger.info(f"🎭 叙事热度: {narrative_scores}")
        return narrative_scores
        
    except Exception as e:
        logger.error(f"获取趋势叙事失败: {e}")
        return {}


def get_narrative_multiplier(inst_id: str, narratives: Dict[str, float]) -> Tuple[float, List[str]]:
    if not narratives or not Config.NARRATIVE_ENABLED:
        return 1.0, []
    
    inst_lower = inst_id.lower().replace("-usdt", "").replace("-usdt-swap", "")
    matched_narratives = []
    max_score = 0.0
    
    narrative_keywords = {
        "ai": ["ai", "agent", "gpt", "llm", "neural", "intelligence", "skyai"],
        "meme": ["meme", "doge", "shib", "pepe", "wif", "floki"],
        "rwa": ["rwa", "realworld", "real world", "tokenized", "land", "property"],
        "gaming": ["game", "gaming", "metaverse", "play", "arena", "hero"],
        "defi": ["defi", "lending", "yield", "swap", "perpetual"],
        "layer2": ["layer2", "l2", "arbitrum", "optimism", "zksync", "polygon"],
        "solana": ["sol", "solana"],
        "mcp": ["mcp", "model context", "protocol"],
    }
    
    for narrative, keywords in narrative_keywords.items():
        for kw in keywords:
            if kw in inst_lower:
                raw_heat = narratives.get(narrative, 0)
                norm_heat = min(1.0, raw_heat / 100.0) if raw_heat > 0 else 0.5
                matched_narratives.append(narrative)
                max_score = max(max_score, norm_heat)
                break
    
    if matched_narratives:
        multiplier = 1.0 + (max_score * (Config.NARRATIVE_WEIGHT - 1.0))
        multiplier = min(2.5, multiplier)
        return multiplier, matched_narratives
    return 1.0, []


# ============================================================
# 做多技术分析
# ============================================================
def analyze_long(df: pd.DataFrame) -> Dict:
    close = df["c"]
    volume = df["vol"]
    high = df["h"]
    
    if len(volume) > 20:
        vol_sma = volume.iloc[:-1].rolling(20).mean().iloc[-1]
        curr_vol = volume.iloc[-1]
        vol_ratio = round(curr_vol / (vol_sma + 1e-9), 2)
        vol_ok = bool(curr_vol > vol_sma * Config.VOL_MULTIPLIER)
    else:
        vol_ratio, vol_ok = 0, False
    
    if len(high) > 20:
        recent_high = high.iloc[-21:-1].max()
        break_ok = bool(close.iloc[-1] > recent_high)
    else:
        break_ok = False
    
    ema50 = calc_ema(close, 50)
    if len(ema50) > 0 and not pd.isna(ema50.iloc[-1]):
        trend_ok = bool(close.iloc[-1] > ema50.iloc[-1])
    else:
        trend_ok = False
    
    macd, signal_line = calc_macd(close)
    if len(macd) > 1 and len(signal_line) > 1:
        macd_cross = bool(
            macd.iloc[-1] > signal_line.iloc[-1] and
            macd.iloc[-2] <= signal_line.iloc[-2]
        )
    else:
        macd_cross = False
    
    rsi = calc_rsi(close, Config.RSI_PERIOD)
    if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
        rsi_val = round(rsi.iloc[-1], 1)
        rsi_ok = bool(Config.LONG_MIN_RSI <= rsi_val <= Config.LONG_MAX_RSI)
    else:
        rsi_val, rsi_ok = 50, False
    
    if len(close) > 1:
        change_15m = round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2)
        change_ok = bool(Config.LONG_MIN_CHANGE_15M <= change_15m <= Config.LONG_MAX_CHANGE_15M)
    else:
        change_15m, change_ok = 0, False
    
    upper, middle, lower = calc_bollinger_bands(close, Config.BB_PERIOD, Config.BB_STD)
    if len(upper) > 0 and not pd.isna(upper.iloc[-1]):
        bb_position = round((close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9), 2)
        bb_ok = bool(0.1 <= bb_position <= 0.8)
    else:
        bb_position, bb_ok = 0, False
    
    k, d, j = calc_kdj(df, Config.KDJ_PERIOD)
    if len(k) > 1 and not pd.isna(k.iloc[-1]):
        kdj_cross = bool(
            k.iloc[-1] > d.iloc[-1] and
            k.iloc[-2] <= d.iloc[-2]
        )
        k_val = round(k.iloc[-1], 1)
        j_val = round(j.iloc[-1], 1)
    else:
        kdj_cross, k_val, j_val = False, 50, 50
    
    kdj_oversold = bool(j_val < 30)
    
    base_score = sum([vol_ok, break_ok, trend_ok, macd_cross, rsi_ok, change_ok, bb_ok, kdj_cross, kdj_oversold])
    
    return {
        "base_score": base_score,
        "vol_ratio": vol_ratio,
        "break_ok": break_ok,
        "trend_ok": trend_ok,
        "macd_cross": macd_cross,
        "rsi": rsi_val,
        "change_15m": change_15m,
        "price": round(close.iloc[-1], 6),
        "bb_position": bb_position,
        "kdj_k": k_val,
        "kdj_j": j_val,
    }


# ============================================================
# 做空技术分析
# ============================================================
def analyze_short(df: pd.DataFrame) -> Dict:
    close = df["c"]
    volume = df["vol"]
    low = df["l"]
    
    if len(volume) > 20:
        vol_sma = volume.iloc[:-1].rolling(20).mean().iloc[-1]
        curr_vol = volume.iloc[-1]
        vol_ratio = round(curr_vol / (vol_sma + 1e-9), 2)
        vol_ok = bool(curr_vol > vol_sma * Config.VOL_MULTIPLIER)
    else:
        vol_ratio, vol_ok = 0, False
    
    if len(low) > 20:
        recent_low = low.iloc[-21:-1].min()
        break_ok = bool(close.iloc[-1] < recent_low)
    else:
        break_ok = False
    
    ema50 = calc_ema(close, 50)
    if len(ema50) > 0 and not pd.isna(ema50.iloc[-1]):
        trend_ok = bool(close.iloc[-1] < ema50.iloc[-1])
    else:
        trend_ok = False
    
    macd, signal_line = calc_macd(close)
    if len(macd) > 1 and len(signal_line) > 1:
        macd_cross = bool(
            macd.iloc[-1] < signal_line.iloc[-1] and
            macd.iloc[-2] >= signal_line.iloc[-2]
        )
    else:
        macd_cross = False
    
    rsi = calc_rsi(close, Config.RSI_PERIOD)
    if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
        rsi_val = round(rsi.iloc[-1], 1)
        rsi_ok = bool(Config.SHORT_MIN_RSI <= rsi_val <= Config.SHORT_MAX_RSI)
    else:
        rsi_val, rsi_ok = 50, False
    
    if len(close) > 1:
        change_15m = round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2)
        change_ok = bool(Config.SHORT_MAX_CHANGE_15M <= change_15m <= Config.SHORT_MIN_CHANGE_15M)
    else:
        change_15m, change_ok = 0, False
    
    upper, middle, lower = calc_bollinger_bands(close, Config.BB_PERIOD, Config.BB_STD)
    if len(upper) > 0 and not pd.isna(upper.iloc[-1]):
        bb_position = round((close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-9), 2)
        bb_ok = bool(bb_position >= 0.7)
    else:
        bb_position, bb_ok = 0, False
    
    k, d, j = calc_kdj(df, Config.KDJ_PERIOD)
    if len(k) > 1 and not pd.isna(k.iloc[-1]):
        kdj_cross = bool(
            k.iloc[-1] < d.iloc[-1] and
            k.iloc[-2] >= d.iloc[-2]
        )
        k_val = round(k.iloc[-1], 1)
        j_val = round(j.iloc[-1], 1)
    else:
        kdj_cross, k_val, j_val = False, 50, 50
    
    kdj_overbought = bool(j_val > 80)
    
    base_score = sum([vol_ok, break_ok, trend_ok, macd_cross, rsi_ok, change_ok, bb_ok, kdj_cross, kdj_overbought])
    
    return {
        "base_score": base_score,
        "vol_ratio": vol_ratio,
        "break_ok": break_ok,
        "trend_ok": trend_ok,
        "macd_cross": macd_cross,
        "rsi": rsi_val,
        "change_15m": change_15m,
        "price": round(close.iloc[-1], 6),
        "bb_position": bb_position,
        "kdj_k": k_val,
        "kdj_j": j_val,
    }


# ============================================================
# 止盈止损计算
# ============================================================
def calculate_long_tp_sl(entry_price: float, volume_ratio: float, score: int) -> Dict:
    sl_pct = Config.LONG_STOP_LOSS_PERCENT
    tp1_pct = Config.LONG_TAKE_PROFIT_1
    tp2_pct = Config.LONG_TAKE_PROFIT_2
    tp3_pct = Config.LONG_TAKE_PROFIT_3
    
    if Config.DYNAMIC_TP:
        if volume_ratio >= 3:
            tp1_pct = Config.LONG_TAKE_PROFIT_1 * 1.2
            tp2_pct = Config.LONG_TAKE_PROFIT_2 * 1.2
            tp3_pct = Config.LONG_TAKE_PROFIT_3 * 1.2
        elif volume_ratio >= 2:
            tp1_pct = Config.LONG_TAKE_PROFIT_1 * 1.1
            tp2_pct = Config.LONG_TAKE_PROFIT_2 * 1.1
            tp3_pct = Config.LONG_TAKE_PROFIT_3 * 1.1
        
        if score >= 7:
            sl_pct = Config.LONG_STOP_LOSS_PERCENT * 1.3
    
    def round_price(p):
        if p < 0.1:
            return round(p, 6)
        elif p < 1:
            return round(p, 4)
        elif p < 100:
            return round(p, 2)
        return round(p, 2)
    
    return {
        'stop_loss': round_price(entry_price * (1 - sl_pct / 100)),
        'stop_loss_pct': sl_pct,
        'take_profit_1': round_price(entry_price * (1 + tp1_pct / 100)),
        'take_profit_1_pct': tp1_pct,
        'take_profit_2': round_price(entry_price * (1 + tp2_pct / 100)),
        'take_profit_2_pct': tp2_pct,
        'take_profit_3': round_price(entry_price * (1 + tp3_pct / 100)),
        'take_profit_3_pct': tp3_pct,
    }


def calculate_short_tp_sl(entry_price: float, volume_ratio: float, score: int) -> Dict:
    sl_pct = Config.SHORT_STOP_LOSS_PERCENT
    tp1_pct = Config.SHORT_TAKE_PROFIT_1
    tp2_pct = Config.SHORT_TAKE_PROFIT_2
    tp3_pct = Config.SHORT_TAKE_PROFIT_3
    
    if Config.DYNAMIC_TP:
        if volume_ratio >= 3:
            tp1_pct = Config.SHORT_TAKE_PROFIT_1 * 1.2
            tp2_pct = Config.SHORT_TAKE_PROFIT_2 * 1.2
            tp3_pct = Config.SHORT_TAKE_PROFIT_3 * 1.2
        elif volume_ratio >= 2:
            tp1_pct = Config.SHORT_TAKE_PROFIT_1 * 1.1
            tp2_pct = Config.SHORT_TAKE_PROFIT_2 * 1.1
            tp3_pct = Config.SHORT_TAKE_PROFIT_3 * 1.1
        
        if score >= 7:
            sl_pct = Config.SHORT_STOP_LOSS_PERCENT * 1.3
    
    def round_price(p):
        if p < 0.1:
            return round(p, 6)
        elif p < 1:
            return round(p, 4)
        elif p < 100:
            return round(p, 2)
        return round(p, 2)
    
    return {
        'stop_loss': round_price(entry_price * (1 + sl_pct / 100)),
        'stop_loss_pct': sl_pct,
        'take_profit_1': round_price(entry_price * (1 - tp1_pct / 100)),
        'take_profit_1_pct': tp1_pct,
        'take_profit_2': round_price(entry_price * (1 - tp2_pct / 100)),
        'take_profit_2_pct': tp2_pct,
        'take_profit_3': round_price(entry_price * (1 - tp3_pct / 100)),
        'take_profit_3_pct': tp3_pct,
    }


# ============================================================
# 信号质量检查（纳入叙事乘数）
# ============================================================
def is_quality_long_signal(details: Dict, narrative_multiplier: float) -> Tuple[bool, str, int]:
    base_score = details["base_score"]
    final_score = int(round(base_score * narrative_multiplier))
    vol_ratio = details["vol_ratio"]
    change_15m = details["change_15m"]
    rsi = details["rsi"]
    
    if vol_ratio <= 1.0:
        return False, f"缩量上涨({vol_ratio}x)", final_score
    if vol_ratio > Config.MAX_VOLUME_RATIO:
        return False, f"放量异常({vol_ratio}x)", final_score
    if rsi > Config.LONG_MAX_RSI:
        return False, f"RSI过高({rsi})", final_score
    if rsi < Config.LONG_MIN_RSI:
        return False, f"RSI过低({rsi})", final_score
    if change_15m < Config.LONG_MIN_CHANGE_15M:
        return False, f"涨幅不足({change_15m}%)", final_score
    
    if final_score >= 6 and vol_ratio >= 1.2:
        return True, f"高分做多信号({final_score}/9)+放量{vol_ratio}x", final_score
    if final_score >= 5 and vol_ratio >= 1.5 and change_15m >= 1.0:
        return True, f"放量上涨({vol_ratio}x,+{change_15m}%)", final_score
    if final_score >= 4 and vol_ratio >= 2.0 and change_15m >= 1.2:
        return True, f"爆量突破({vol_ratio}x,+{change_15m}%)", final_score
    if vol_ratio >= 2.5 and change_15m >= 0.8 and details["trend_ok"]:
        return True, f"爆量启动({vol_ratio}x,+{change_15m}%)", final_score
    
    return False, f"条件不足(得分{final_score}/9,放量{vol_ratio}x,涨幅{change_15m}%)", final_score


def is_quality_short_signal(details: Dict, narrative_multiplier: float) -> Tuple[bool, str, int]:
    base_score = details["base_score"]
    final_score = int(round(base_score * narrative_multiplier))
    vol_ratio = details["vol_ratio"]
    change_15m = details["change_15m"]
    rsi = details["rsi"]
    
    if vol_ratio <= 1.0:
        return False, f"缩量下跌({vol_ratio}x)", final_score
    if vol_ratio > Config.MAX_VOLUME_RATIO:
        return False, f"放量异常({vol_ratio}x)", final_score
    if rsi < Config.SHORT_MIN_RSI:
        return False, f"RSI不够高({rsi} < {Config.SHORT_MIN_RSI})", final_score
    if rsi > Config.SHORT_MAX_RSI:
        return False, f"RSI过高({rsi})", final_score
    if change_15m > Config.SHORT_MAX_CHANGE_15M:
        return False, f"跌幅不足({change_15m}%)", final_score
    
    if final_score >= 6 and vol_ratio >= 1.2:
        return True, f"高分做空信号({final_score}/9)+放量{vol_ratio}x", final_score
    if final_score >= 4 and vol_ratio >= 1.5 and change_15m <= -1.0:
        return True, f"放量下跌({vol_ratio}x,{change_15m}%)", final_score
    if final_score >= 4 and details["break_ok"] and vol_ratio >= 1.5:
        return True, f"跌破支撑+放量{vol_ratio}x", final_score
    if vol_ratio >= 2.5 and change_15m <= -1.0 and details["trend_ok"]:
        return True, f"爆量下跌({vol_ratio}x,{change_15m}%)", final_score
    
    return False, f"条件不足(得分{final_score}/9,放量{vol_ratio}x,跌幅{change_15m}%)", final_score


# ============================================================
# 信号缓存
# ============================================================
class SignalCache:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
    
    def should_send(self, inst_id: str, score: int, price: float) -> bool:
        now = time.time()
        cached = self._cache.get(inst_id)
        
        if not cached:
            return True
        
        if now - cached["timestamp"] < Config.SIGNAL_COOLDOWN:
            if score - cached["score"] >= Config.SCORE_UPGRADE_THRESHOLD:
                return True
            return False
        return True
    
    def update(self, inst_id: str, score: int, price: float):
        self._cache[inst_id] = {
            "timestamp": time.time(),
            "score": score,
            "price": price
        }


signal_cache = SignalCache()


# ============================================================
# 信号分析（集成叙事热度）
# ============================================================
def analyze_long_symbol(inst_id: str, change_24h: float, narratives: Dict[str, float]) -> Optional[Dict]:
    if not signal_cache.should_send(inst_id, 999, 0):
        return None
    
    df = get_klines(inst_id)
    if df is None:
        return None
    
    tech_details = analyze_long(df)
    base_score = tech_details["base_score"]
    
    narrative_multiplier, matched_narratives = get_narrative_multiplier(inst_id, narratives)
    final_score = int(round(base_score * narrative_multiplier))
    
    is_quality, reason, final_score_calc = is_quality_long_signal(tech_details, narrative_multiplier)
    
    if is_quality and final_score_calc >= Config.LONG_SCORE_THRESHOLD:
        tp_sl = calculate_long_tp_sl(tech_details["price"], tech_details["vol_ratio"], final_score_calc)
        return {
            "inst_id": inst_id,
            "direction": "LONG",
            "details": tech_details,
            "final_score": final_score_calc,
            "base_score": base_score,
            "narrative_multiplier": narrative_multiplier,
            "matched_narratives": matched_narratives,
            "change_24h": change_24h,
            "quality_reason": reason,
            "tp_sl": tp_sl,
        }
    return None


def analyze_short_symbol(inst_id: str, change_24h: float, narratives: Dict[str, float]) -> Optional[Dict]:
    if not signal_cache.should_send(inst_id, 999, 0):
        return None
    
    df = get_klines(inst_id)
    if df is None:
        return None
    
    tech_details = analyze_short(df)
    base_score = tech_details["base_score"]
    
    narrative_multiplier, matched_narratives = get_narrative_multiplier(inst_id, narratives)
    final_score = int(round(base_score * narrative_multiplier))
    
    is_quality, reason, final_score_calc = is_quality_short_signal(tech_details, narrative_multiplier)
    
    if is_quality and final_score_calc >= Config.SHORT_SCORE_THRESHOLD:
        tp_sl = calculate_short_tp_sl(tech_details["price"], tech_details["vol_ratio"], final_score_calc)
        return {
            "inst_id": inst_id,
            "direction": "SHORT",
            "details": tech_details,
            "final_score": final_score_calc,
            "base_score": base_score,
            "narrative_multiplier": narrative_multiplier,
            "matched_narratives": matched_narratives,
            "change_24h": change_24h,
            "quality_reason": reason,
            "tp_sl": tp_sl,
        }
    return None


# ============================================================
# DeepSeek 分析模块（加入暴涨可能性评估）
# ============================================================
def deepseek_analyze_signal(signal: Dict) -> str:
    """
    调用 DeepSeek API 分析单个信号，返回分析评语（含暴涨可能性）。
    若失败或未启用，返回空字符串。
    """
    if not Config.ENABLE_DEEPSEEK or not Config.DEEPSEEK_API_KEY:
        return ""

    d = signal["details"]
    direction = signal["direction"]

    prompt = f"""你是一个加密货币量化交易分析师。请对以下交易信号进行简短分析，并给出**暴涨可能性（高/中/低）**。

币种: {signal['inst_id']}
方向: {direction}
当前价格: {d['price']}
15分钟涨跌幅: {d['change_15m']:.2f}%
24小时涨跌幅: {signal['change_24h']}%
成交量比(15m): {d['vol_ratio']}x
RSI(14): {d['rsi']}
技术得分: {signal['base_score']}/9
叙事乘数: {signal['narrative_multiplier']:.2f}x
最终得分: {signal['final_score']}/9
信号理由: {signal['quality_reason']}

请用一句话输出（不超过60字），格式为：“分析内容。暴涨可能性：高/中/低”。"""

    headers = {
        "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": Config.DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 120
    }

    try:
        resp = requests.post(
            f"{Config.DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=Config.DEEPSEEK_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        analysis = data["choices"][0]["message"]["content"].strip()
        return f"\n🤖 **DeepSeek分析:** {analysis}"
    except Exception as e:
        logger.warning(f"DeepSeek 分析失败: {e}")
        return ""


# ============================================================
# 飞书推送（增强叙事显示 + DeepSeek）
# ============================================================
def send_feishu(signals: List[Dict]) -> bool:
    if not Config.FEISHU_WEBHOOK or not signals:
        return False
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    long_signals = [s for s in signals if s["direction"] == "LONG"]
    short_signals = [s for s in signals if s["direction"] == "SHORT"]
    
    content_lines = [f"🚀 **多空信号扫描** | {now}\n"]
    content_lines.append(f"📊 做多: {len(long_signals)}个 | 做空: {len(short_signals)}个\n")
    content_lines.append("━━━━━━━━━━━━━━━━━━━━\n")
    
    for s in signals:
        d = s["details"]
        tp_sl = s["tp_sl"]
        
        if s["direction"] == "LONG":
            direction_emoji = "🟢"
            direction_text = "做多"
            if s["final_score"] >= 7:
                advice = "🔥🔥 强力买入"
            elif s["final_score"] >= 5:
                advice = "✅ 建议买入"
            else:
                advice = "👀 关注"
        else:
            direction_emoji = "🔴"
            direction_text = "做空"
            if s["final_score"] >= 7:
                advice = "🔥🔥 强力做空"
            elif s["final_score"] >= 5:
                advice = "✅ 建议做空"
            else:
                advice = "👀 关注做空"
        
        if d["vol_ratio"] >= 3:
            mark = "💥💥"
        elif d["vol_ratio"] >= 2:
            mark = "💥"
        else:
            mark = "🔥"
        
        msg = f"{direction_emoji} {mark} **{s['inst_id']}** | {direction_text} | {advice}\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"💰 **入场价格:** {d['price']}\n"
        change_emoji = "📈" if d['change_15m'] > 0 else "📉"
        msg += f"{change_emoji} **15m变化:** {d['change_15m']:+.2f}% | **24h涨幅:** {s['change_24h']}%\n"
        msg += f"⭐ **最终得分:** {s['final_score']}/9 | 技术分: {s['base_score']}/9 | 叙事乘数: {s['narrative_multiplier']:.2f}x\n"
        if s["matched_narratives"]:
            msg += f"🎭 **匹配叙事:** {', '.join(s['matched_narratives'])}\n"
        msg += f"📝 **信号理由:** {s['quality_reason']}\n\n"
        
        # DeepSeek 分析结果（如果存在）
        if s.get("deepseek_analysis"):
            msg += f"{s['deepseek_analysis']}\n"
        
        if s["direction"] == "LONG":
            msg += f"🎯 **止盈止损位 (做多):**\n"
            msg += f"  ├─ 🛑 **止损:** {tp_sl['stop_loss']} (-{tp_sl['stop_loss_pct']:.1f}%)\n"
            msg += f"  ├─ 🎯 **TP1:** {tp_sl['take_profit_1']} (+{tp_sl['take_profit_1_pct']:.1f}%)\n"
            msg += f"  ├─ 🎯 **TP2:** {tp_sl['take_profit_2']} (+{tp_sl['take_profit_2_pct']:.1f}%)\n"
            msg += f"  └─ 🎯 **TP3:** {tp_sl['take_profit_3']} (+{tp_sl['take_profit_3_pct']:.1f}%)\n"
        else:
            msg += f"🎯 **止盈止损位 (做空):**\n"
            msg += f"  ├─ 🛑 **止损:** {tp_sl['stop_loss']} (+{tp_sl['stop_loss_pct']:.1f}%)\n"
            msg += f"  ├─ 🎯 **TP1:** {tp_sl['take_profit_1']} (-{tp_sl['take_profit_1_pct']:.1f}%)\n"
            msg += f"  ├─ 🎯 **TP2:** {tp_sl['take_profit_2']} (-{tp_sl['take_profit_2_pct']:.1f}%)\n"
            msg += f"  └─ 🎯 **TP3:** {tp_sl['take_profit_3']} (-{tp_sl['take_profit_3_pct']:.1f}%)\n"
        
        msg += f"\n📊 **技术指标:**\n"
        trend_text = "✅向上" if d['trend_ok'] and s["direction"] == "LONG" else ("❌向下" if d['trend_ok'] and s["direction"] == "SHORT" else "➡️震荡")
        msg += f"  ├─ 📈 趋势: {trend_text}\n"
        break_text = "✅突破高点" if d.get('break_ok') and s["direction"] == "LONG" else ("✅跌破低点" if d.get('break_ok') and s["direction"] == "SHORT" else "❌未突破")
        msg += f"  ├─ 🔓 形态: {break_text}\n"
        msg += f"  ├─ 📊 布林带位置: {d['bb_position']:.2f}\n"
        
        if d['kdj_j'] > 80:
            kdj_status = f"⚠️超买 J={d['kdj_j']:.1f}"
        elif d['kdj_j'] < 20:
            kdj_status = f"✅超卖 J={d['kdj_j']:.1f}"
        else:
            kdj_status = f"K={d['kdj_k']:.1f} J={d['kdj_j']:.1f}"
        msg += f"  └─ 📉 KDJ: {kdj_status}\n"
        
        msg += f"\n⏰ 将在1h/4h/24h后验证信号表现\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        
        content_lines.append(msg)
    
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": f"🚀 多空信号 | 做多{len(long_signals)} 做空{len(short_signals)}",
                    "content": [[{"tag": "text", "text": "\n".join(content_lines)}]]
                }
            }
        }
    }
    
    try:
        requests.post(Config.FEISHU_WEBHOOK, json=payload, timeout=10)
        logger.info(f"飞书推送成功: 做多{len(long_signals)}个 做空{len(short_signals)}个")
        return True
    except Exception as e:
        logger.error(f"飞书推送失败: {e}")
        return False


# ============================================================
# 验证功能
# ============================================================
def verify_signals():
    logger.info("开始验证待处理信号...")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM signal_verifications
            WHERE status = 'pending' AND verify_time <= datetime('now')
            ORDER BY verify_time ASC
            LIMIT 100
        """)
        pending = cursor.fetchall()
    
    if not pending:
        return
    
    logger.info(f"发现 {len(pending)} 个待验证信号")
    
    for task in pending:
        inst_id = task['inst_id']
        signal_price = task['signal_price']
        verify_hours = task['verify_hours']
        direction = task['direction']
        
        current_price = get_current_price(inst_id)
        
        if current_price and current_price > 0:
            if direction == "LONG":
                change_percent = (current_price - signal_price) / signal_price * 100
            else:
                change_percent = (signal_price - current_price) / signal_price * 100
            
            change_percent = round(change_percent, 2)
            
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE signal_verifications
                    SET verify_price = ?, change_percent = ?, status = 'completed', verify_time = ?
                    WHERE id = ?
                """, (current_price, change_percent, datetime.now(), task['id']))
                conn.commit()
            
            emoji = "✅" if change_percent > 0 else "❌"
            logger.info(f"{emoji} 验证 {inst_id} | {direction} | {verify_hours}h | 收益: {change_percent:+.2f}%")
        else:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE signal_verifications
                    SET status = 'failed', verify_time = ?
                    WHERE id = ?
                """, (datetime.now(), task['id']))
                conn.commit()


# ============================================================
# 市场监控
# ============================================================
def get_market_health() -> dict:
    try:
        btc_ticker = get_ticker("BTC-USDT")
        btc_volume = 0
        btc_change = 0
        
        if btc_ticker:
            btc_volume = float(btc_ticker.get("volCcy24h", 0)) / 1e6
            btc_change = float(btc_ticker.get("chg24h", 0))
            if 0 < abs(btc_change) < 1:
                btc_change = btc_change * 100
        
        if btc_volume > 2000:
            activity = "🔥🔥🔥 高度活跃"
        elif btc_volume > 1000:
            activity = "🔥🔥 中度活跃"
        elif btc_volume > 500:
            activity = "🔥 低度活跃"
        else:
            activity = "❄️ 冷清"
        
        return {
            'btc_volume': round(btc_volume, 2),
            'btc_change': round(btc_change, 2),
            'activity': activity,
        }
    except Exception as e:
        logger.error(f"获取市场健康数据失败: {e}")
        return None


def send_market_report():
    if not Config.FEISHU_WEBHOOK:
        return
    
    market = get_market_health()
    if not market:
        return
    
    stats = db.get_stats()
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content_lines = [f"📊 **市场统计报告** | {now}\n"]
    content_lines.append("━━━━━━━━━━━━━━━━━━━━\n")
    content_lines.append(f"**市场活跃度:** {market['activity']}\n")
    content_lines.append(f"\n**主流币种:**\n")
    content_lines.append(f"  • BTC: ${market['btc_volume']:.0f}M | 24h: {market['btc_change']:+.2f}%\n")
    content_lines.append(f"\n**信号统计:**\n")
    content_lines.append(f"  • 总信号数: {stats['total']}\n")
    content_lines.append(f"  • 做多信号: {stats['long']}\n")
    content_lines.append(f"  • 做空信号: {stats['short']}\n")
    content_lines.append("\n━━━━━━━━━━━━━━━━━━━━\n")
    
    payload = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": f"📊 市场统计报告",
                    "content": [[{"tag": "text", "text": "\n".join(content_lines)}]]
                }
            }
        }
    }
    
    try:
        requests.post(Config.FEISHU_WEBHOOK, json=payload, timeout=10)
        logger.info("市场报告推送成功")
    except Exception as e:
        logger.error(f"市场报告推送失败: {e}")


# ============================================================
# 扫描主函数（集成叙事 + DeepSeek）
# ============================================================
def scan() -> None:
    logger.info("=" * 50)
    direction_text = "做多" + ("+做空" if Config.ENABLE_SHORT else "")
    logger.info(f"开始新一轮扫描 - {direction_text}")
    start_time = time.time()
    
    narratives = get_top_narratives() if Config.NARRATIVE_ENABLED else {}
    
    try:
        spot_symbols = get_spot_symbols()
        swap_symbols = get_swap_symbols()
        all_symbols = spot_symbols + swap_symbols
        logger.info(f"获取到 {len(all_symbols)} 个交易对")
        
        filtered = []
        for inst_id in all_symbols:
            ticker = get_ticker(inst_id)
            if not ticker:
                continue
            
            vol_usd = float(ticker.get("volCcy24h", 0))
            if vol_usd == 0 and ticker.get("vol24h") and ticker.get("last"):
                vol_usd = float(ticker.get("vol24h", 0)) * float(ticker.get("last", 0))
            
            if vol_usd >= Config.MIN_VOLUME_USD:
                change_raw = ticker.get("chg24h", "0")
                try:
                    change_raw = float(change_raw)
                    if 0 < abs(change_raw) < 1:
                        change_24h = round(change_raw * 100, 2)
                    else:
                        change_24h = round(change_raw, 2)
                except:
                    change_24h = 0
                filtered.append({"inst_id": inst_id, "change_24h": change_24h})
        
        logger.info(f"成交额过滤后: {len(filtered)} 个")
        
        if not filtered:
            logger.info("无符合条件的币种")
            return
        
        triggers = []
        
        with ThreadPoolExecutor(max_workers=min(Config.CONCURRENT_WORKERS, len(filtered) * 2)) as executor:
            futures = []
            for item in filtered:
                futures.append(executor.submit(analyze_long_symbol, item["inst_id"], item["change_24h"], narratives))
            if Config.ENABLE_SHORT:
                for item in filtered:
                    futures.append(executor.submit(analyze_short_symbol, item["inst_id"], item["change_24h"], narratives))
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    triggers.append(result)
                    signal_cache.update(
                        result["inst_id"], 
                        result["final_score"],
                        result["details"]["price"]
                    )
                    d = result["details"]
                    direction_emoji = "🟢" if result["direction"] == "LONG" else "🔴"
                    logger.info(
                        f"{direction_emoji} {result['inst_id']} | {result['direction']} | {result['quality_reason']} | "
                        f"最终分{result['final_score']}/9 | 技术分{d['base_score']}/9 | 叙事乘数{result['narrative_multiplier']:.2f}x | "
                        f"放量{d['vol_ratio']}x | 15m:{d['change_15m']:+.2f}% | RSI{d['rsi']}"
                    )
                    
                    signal_id = db.save_signal({
                        'inst_id': result["inst_id"],
                        'direction': result["direction"],
                        'signal_time': datetime.now(),
                        'price': d["price"],
                        'score': result["final_score"],
                        'narrative_multiplier': result["narrative_multiplier"],
                        'volume_ratio': d["vol_ratio"],
                        'change_15m': d["change_15m"],
                        'change_24h': result["change_24h"],
                        'rsi': d["rsi"],
                        'bb_position': d.get("bb_position"),
                        'kdj_k': d.get("kdj_k"),
                        'kdj_d': d.get("kdj_d"),
                        'kdj_j': d.get("kdj_j"),
                        'quality_reason': result["quality_reason"],
                        'stop_loss': result["tp_sl"]['stop_loss'],
                        'take_profit_1': result["tp_sl"]['take_profit_1'],
                        'take_profit_2': result["tp_sl"]['take_profit_2'],
                        'take_profit_3': result["tp_sl"]['take_profit_3']
                    })
                    
                    if signal_id:
                        db.add_verification_tasks(
                            signal_id, 
                            result["inst_id"], 
                            result["direction"],
                            datetime.now(), 
                            d["price"]
                        )
        
        elapsed = time.time() - start_time
        long_count = len([t for t in triggers if t["direction"] == "LONG"])
        short_count = len([t for t in triggers if t["direction"] == "SHORT"])
        logger.info(f"扫描完成，耗时 {elapsed:.1f} 秒 | 做多:{long_count}个 做空:{short_count}个")
        
        if triggers:
            if Config.ENABLE_DEEPSEEK:
                logger.info("正在调用 DeepSeek 分析信号...")
                with ThreadPoolExecutor(max_workers=5) as deepseek_executor:
                    deepseek_futures = {deepseek_executor.submit(deepseek_analyze_signal, s): s for s in triggers}
                    for future in as_completed(deepseek_futures):
                        signal = deepseek_futures[future]
                        signal["deepseek_analysis"] = future.result()
            send_feishu(triggers)
        
    except Exception as e:
        logger.exception("扫描异常")


# ============================================================
# 健康检查
# ============================================================
def health_check():
    try:
        ticker = get_ticker("BTC-USDT")
        if ticker:
            stats = db.get_stats()
            logger.info(f"健康检查通过 - 总信号:{stats['total']}个, 做多:{stats['long']}, 做空:{stats['short']}")
            
            if Config.FEISHU_WEBHOOK:
                market = get_market_health()
                if market:
                    msg = f"✅ 多空扫描器已启动\n"
                    msg += f"📊 市场: {market['activity']}\n"
                    msg += f"💰 BTC: ${market['btc_volume']:.0f}M | {market['btc_change']:+.2f}%\n"
                    msg += f"🎯 做多阈值: 最终得分≥{Config.LONG_SCORE_THRESHOLD} | 放量≥{Config.LONG_MIN_VOLUME_RATIO}x\n"
                    msg += f"🎯 做空阈值: 最终得分≥{Config.SHORT_SCORE_THRESHOLD} | 放量≥{Config.SHORT_MIN_VOLUME_RATIO}x"
                    if Config.NARRATIVE_ENABLED:
                        msg += f"\n🎭 叙事热度已启用 (CoinGecko Trending) | 权重系数: {Config.NARRATIVE_WEIGHT}x"
                    if Config.ENABLE_DEEPSEEK:
                        msg += f"\n🤖 DeepSeek 分析已启用 (含暴涨可能性评估)"
                    
                    payload = {"msg_type": "text", "content": {"text": msg}}
                    requests.post(Config.FEISHU_WEBHOOK, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"健康检查失败: {e}")


# ============================================================
# 主函数
# ============================================================
def main():
    db_dir = os.path.dirname(Config.DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("OKX 多空信号扫描器启动 v3.2 (叙事热度 + DeepSeek 暴涨可能性)")
    logger.info(f"做多: 最终得分≥{Config.LONG_SCORE_THRESHOLD} | 放量≥{Config.LONG_MIN_VOLUME_RATIO}x | 涨幅≥{Config.LONG_MIN_CHANGE_15M}%")
    logger.info(f"做空: 最终得分≥{Config.SHORT_SCORE_THRESHOLD} | 放量≥{Config.SHORT_MIN_VOLUME_RATIO}x | 跌幅≤{Config.SHORT_MIN_CHANGE_15M}%")
    if Config.NARRATIVE_ENABLED:
        logger.info(f"叙事热度: 已启用 | 权重系数: {Config.NARRATIVE_WEIGHT}x | 数据源: CoinGecko Trending API")
    else:
        logger.info("叙事热度: 已禁用")
    if Config.ENABLE_DEEPSEEK:
        logger.info(f"DeepSeek 分析: 已启用 | 模型: {Config.DEEPSEEK_MODEL} (含暴涨可能性评估)")
    else:
        logger.info("DeepSeek 分析: 已禁用")
    
    health_check()
    
    stats = db.get_stats()
    logger.info(f"历史统计: 总信号{stats['total']}个 | 做多:{stats['long']} | 做空:{stats['short']}")
    
    scan()
    
    schedule.every(Config.SCAN_INTERVAL).minutes.do(scan)
    schedule.every(Config.VERIFY_CHECK_INTERVAL).minutes.do(verify_signals)
    
    for hour in [8, 12, 18, 22]:
        schedule.every().day.at(f"{hour:02d}:00").do(send_market_report)
    
    logger.info(f"定时任务已设置:")
    logger.info(f"  - 多空扫描: 每{Config.SCAN_INTERVAL}分钟")
    logger.info(f"  - 信号验证: 每{Config.VERIFY_CHECK_INTERVAL}分钟")
    logger.info(f"  - 市场报告: 每天 8,12,18,22 点")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("正在关闭...")


if __name__ == "__main__":
    main()