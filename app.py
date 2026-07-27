# -*- coding: utf-8 -*-
"""
台股持股警訊監控器（雲端版 v5 瀏覽器本機儲存版）
用途：買進後，依盤中量價與盤後技術指標監控是否出現重大警訊，輔助判斷續抱、留意、觀察、減碼或賣出警訊。
新增：持股資料會儲存在目前瀏覽器的 localStorage，重新開啟同一網址時會自動帶回。

部署：Streamlit Community Cloud
主程式：app.py
"""

from __future__ import annotations

import math
import json
import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:
    streamlit_js_eval = None


# -----------------------------
# Streamlit 基本設定
# -----------------------------
st.set_page_config(
    page_title="台股持股警訊監控器 v5",
    page_icon="🚨",
    layout="wide",
)

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
}

STOCK_NAME_FALLBACK = {
    "1101": "台泥", "1102": "亞泥", "1216": "統一", "1301": "台塑", "1303": "南亞",
    "1313": "聯成", "1326": "台化", "1402": "遠東新", "1476": "儒鴻", "1504": "東元",
    "1605": "華新", "2002": "中鋼", "2105": "正新", "2201": "裕隆", "2301": "光寶科",
    "2303": "聯電", "2308": "台達電", "2317": "鴻海", "2327": "國巨", "2330": "台積電",
    "2352": "佳世達", "2353": "宏碁", "2354": "鴻準", "2356": "英業達", "2357": "華碩",
    "2371": "大同", "2376": "技嘉", "2377": "微星", "2382": "廣達", "2395": "研華",
    "2408": "南亞科", "2409": "友達", "2412": "中華電", "2454": "聯發科", "2474": "可成",
    "2603": "長榮", "2609": "陽明", "2610": "華航", "2615": "萬海", "2618": "長榮航",
    "2881": "富邦金", "2882": "國泰金", "2884": "玉山金", "2885": "元大金", "2891": "中信金",
    "3008": "大立光", "3034": "聯詠", "3035": "智原", "3231": "緯創", "3443": "創意",
    "3481": "群創", "3711": "日月光投控", "4904": "遠傳", "5871": "中租-KY", "5880": "合庫金",
    "6446": "藥華藥", "6505": "台塑化", "6669": "緯穎", "6770": "力積電",
    "1718": "中纖", "1314": "中石化", "1717": "長興", "1722": "台肥", "1802": "台玻",
    "1907": "永豐餘", "2014": "中鴻", "2027": "大成鋼", "2049": "上銀", "2344": "華邦電",
    "2345": "智邦", "2368": "金像電", "2383": "台光電", "2449": "京元電子", "3017": "奇鋐",
    "3023": "信邦", "3036": "文曄", "3044": "健鼎", "3189": "景碩", "3406": "玉晶光",
    "3661": "世芯-KY", "4958": "臻鼎-KY", "5269": "祥碩", "6239": "力成", "6285": "啟碁",
    "6415": "矽力*-KY", "6515": "穎崴", "6781": "AES-KY", "8046": "南電", "8996": "高力",
    "3105": "穩懋", "3260": "威剛", "3324": "雙鴻", "4743": "合一", "5347": "世界",
    "5371": "中光電", "5483": "中美晶", "6187": "萬潤", "6488": "環球晶", "8069": "元太",
    "8086": "宏捷科", "8299": "群聯",
}

BROWSER_STORAGE_KEY = "tw_holding_risk_monitor_v5_holdings"

DEFAULT_HOLDING_COLUMNS = [
    "code", "name", "buy_date", "buy_price", "quantity", "stop_loss", "entry_score",
    "entry_reasons", "note", "status", "sell_date", "sell_price", "sell_reason"
]


# -----------------------------
# 小工具
# -----------------------------
def clean_code(raw: str) -> str:
    return str(raw).strip().upper().replace(" ", "")


def code_to_ticker(raw: str, market_hint: str = "") -> str:
    c = clean_code(raw)
    if c.endswith(".TW") or c.endswith(".TWO"):
        return c
    if "上櫃" in str(market_hint):
        return f"{c}.TWO"
    return f"{c}.TW"


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            s = value.strip().replace(",", "")
            if s in ["", "--", "-", "nan", "None", "null", "N/A"]:
                return default
            return float(s)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fmt_price(v) -> str:
    x = safe_float(v)
    if math.isnan(x):
        return "-"
    return f"{x:,.2f}"


def fmt_pct(v) -> str:
    x = safe_float(v)
    if math.isnan(x):
        return "-"
    return f"{x:+.2f}%"


def normalize_holdings_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=DEFAULT_HOLDING_COLUMNS)
    df = df.copy()
    text_cols = ["code", "name", "buy_date", "entry_reasons", "note", "status", "sell_date", "sell_reason"]
    num_cols = ["buy_price", "quantity", "stop_loss", "entry_score", "sell_price"]
    for c in DEFAULT_HOLDING_COLUMNS:
        if c not in df.columns:
            df[c] = "" if c in text_cols else 0
    df = df[DEFAULT_HOLDING_COLUMNS]
    for c in text_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": "", "NaT": "", "None": ""})
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df.loc[df["status"].eq(""), "status"] = "監控中"
    df = df[df["code"].astype(str).str.strip() != ""].reset_index(drop=True)
    return df


def load_uploaded_holdings(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=DEFAULT_HOLDING_COLUMNS)
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype={"code": str})
        else:
            df = pd.read_excel(uploaded_file, dtype={"code": str})
        return normalize_holdings_df(df)
    except Exception as e:
        st.error(f"讀取持股檔失敗：{e}")
        return pd.DataFrame(columns=DEFAULT_HOLDING_COLUMNS)


def _pick_first_existing(row: Dict, keys: List[str]) -> str:
    """從不同公開資料欄位格式中，取出第一個存在且非空白的值。"""
    for k in keys:
        if k in row and str(row.get(k, "")).strip():
            return str(row.get(k, "")).strip()
    return ""


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_public_stock_name_map() -> Dict[str, str]:
    """抓上市/上櫃公開資料的股票代號與中文名稱。

    不同來源欄位名稱可能不同，所以用多組欄位名稱做彈性判讀。
    若雲端服務或網路暫時抓不到，仍會回傳內建備援名稱表。
    """
    name_map: Dict[str, str] = dict(STOCK_NAME_FALLBACK)
    endpoints = [
        # TWSE 上市全市場日行情，常見欄位：Code、Name
        "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
        # TPEx 上櫃公開資料，欄位可能隨資料集版本不同而有差異
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes",
        "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes",
    ]
    code_keys = [
        "Code", "code", "證券代號", "股票代號", "有價證券代號", "SecuritiesCompanyCode",
        "CompanyCode", "SecurityCode", "SecuritiesCode", "股票證券代號",
    ]
    name_keys = [
        "Name", "name", "證券名稱", "股票名稱", "有價證券名稱", "CompanyName",
        "SecurityName", "SecuritiesName", "公司名稱", "簡稱",
    ]
    for url in endpoints:
        try:
            r = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            data = r.json()
            if isinstance(data, dict):
                # 有些 OpenAPI 會包在 data/result 裡
                for key in ["data", "result", "items"]:
                    if isinstance(data.get(key), list):
                        data = data[key]
                        break
            if not isinstance(data, list):
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                code = _pick_first_existing(row, code_keys)
                name = _pick_first_existing(row, name_keys)
                code = clean_code(code).split(".")[0]
                if code.isdigit() and len(code) == 4 and name:
                    # 排除太不像中文股票名的值，但保留 KY、* 等合法名稱字元
                    name_map[code] = name.strip()
        except Exception:
            continue
    return name_map


def resolve_stock_name(code: str) -> str:
    c = clean_code(code).split(".")[0]
    if not c:
        return ""
    name_map = fetch_public_stock_name_map()
    return name_map.get(c, STOCK_NAME_FALLBACK.get(c, ""))


def autofill_holding_names(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_holdings_df(df)
    if out.empty:
        return out
    for idx, row in out.iterrows():
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        if code and (not name or name.lower() == "nan"):
            out.at[idx, "name"] = resolve_stock_name(code)
    return out


# -----------------------------
# 資料抓取
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_history(ticker: str, hist_range: str = "6mo", interval: str = "1d") -> Tuple[Optional[pd.DataFrame], Dict]:
    ticker = clean_code(ticker)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": hist_range, "interval": interval, "includePrePost": "false", "events": "history"}
    try:
        r = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=12)
        if r.status_code != 200:
            return None, {"error": f"HTTP {r.status_code}", "ticker": ticker}
        data = r.json()
    except Exception as e:
        return None, {"error": f"連線/解析失敗：{e}", "ticker": ticker}

    chart = data.get("chart", {})
    if chart.get("error"):
        return None, {"error": str(chart.get("error")), "ticker": ticker}
    result = (chart.get("result") or [])
    if not result:
        return None, {"error": "無 chart result", "ticker": ticker}
    result = result[0]
    ts = result.get("timestamp") or []
    quote_list = result.get("indicators", {}).get("quote", [])
    if not ts or not quote_list:
        return None, {"error": "無價格資料", "ticker": ticker}

    q = quote_list[0]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Taipei"),
        "open": q.get("open"),
        "high": q.get("high"),
        "low": q.get("low"),
        "close": q.get("close"),
        "volume": q.get("volume"),
    })
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df = df[df["volume"].fillna(0) >= 0]
    if df.empty:
        return None, {"error": "清理後無資料", "ticker": ticker}
    meta = result.get("meta", {}) or {}
    meta["ticker"] = ticker
    return df.reset_index(drop=True), meta


def fetch_with_fallback(code: str, market_hint: str = "", hist_range: str = "6mo") -> Tuple[Optional[pd.DataFrame], Dict]:
    ticker = code_to_ticker(code, market_hint)
    base = ticker.split(".")[0]
    candidates = [ticker]
    other = f"{base}.TWO" if ticker.endswith(".TW") else f"{base}.TW"
    if other not in candidates:
        candidates.append(other)

    last_meta = {}
    for t in candidates:
        df, meta = fetch_yahoo_history(t, hist_range=hist_range, interval="1d")
        if df is not None and len(df) >= 35:
            meta = dict(meta)
            meta["resolved_ticker"] = t
            return df, meta
        last_meta = meta
    return None, last_meta or {"error": "抓取失敗"}


# -----------------------------
# 技術指標
# -----------------------------
def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    close = d["close"]
    high = d["high"]
    low = d["low"]
    open_ = d["open"]
    vol = d["volume"].fillna(0)

    d["ma5"] = close.rolling(5).mean()
    d["ma10"] = close.rolling(10).mean()
    d["vol_ma5"] = vol.rolling(5).mean()
    d["vol_ma20"] = vol.rolling(20).mean()

    # KD/KDJ
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9) * 100
    rsv = rsv.replace([np.inf, -np.inf], np.nan).fillna(50)
    d["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d["D"] = d["K"].ewm(alpha=1 / 3, adjust=False).mean()
    d["J"] = 3 * d["K"] - 2 * d["D"]

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    d["RSI"] = d["RSI"].fillna(100).clip(0, 100)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    d["DIF"] = ema12 - ema26
    d["DEA"] = d["DIF"].ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = d["DIF"] - d["DEA"]

    # CCI 20
    tp = (high + low + close) / 3
    ma_tp = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d["CCI"] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))

    # ADX / DI
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=d.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=d.index)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr.replace(0, np.nan)
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["plus_DI"] = plus_di
    d["minus_DI"] = minus_di
    d["ADX"] = wilder_smooth(dx, 14)

    # K棒型態
    rng = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    upper = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower = pd.concat([open_, close], axis=1).min(axis=1) - low
    d["upper_shadow_ratio"] = (upper / rng).fillna(0)
    d["lower_shadow_ratio"] = (lower / rng).fillna(0)
    d["body_ratio"] = (body / rng).fillna(0)
    d["is_red_k"] = close > open_

    # MTM / ROC
    d["MTM"] = close - close.shift(10)
    d["ROC"] = (close / close.shift(10) - 1) * 100

    # A/DI
    hl_range = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / hl_range).replace([np.inf, -np.inf], np.nan).fillna(0)
    mfv = mfm * vol
    d["ADI"] = mfv.cumsum()

    # BR 26
    prev_close = close.shift(1)
    br_up = (high - prev_close).clip(lower=0)
    br_dn = (prev_close - low).clip(lower=0)
    d["BR"] = br_up.rolling(26).sum() / br_dn.rolling(26).sum().replace(0, np.nan) * 100

    # Bollinger %B
    boll_mid = close.rolling(20).mean()
    boll_std = close.rolling(20).std()
    boll_up = boll_mid + 2 * boll_std
    boll_dn = boll_mid - 2 * boll_std
    d["PCT_B"] = (close - boll_dn) / (boll_up - boll_dn).replace(0, np.nan)

    # VR 26
    up_vol = pd.Series(np.where(close > prev_close, vol, 0.0), index=d.index)
    down_vol = pd.Series(np.where(close < prev_close, vol, 0.0), index=d.index)
    flat_vol = pd.Series(np.where(close == prev_close, vol, 0.0), index=d.index)
    d["VR"] = (up_vol.rolling(26).sum() + 0.5 * flat_vol.rolling(26).sum()) / \
              (down_vol.rolling(26).sum() + 0.5 * flat_vol.rolling(26).sum()).replace(0, np.nan) * 100

    # EMV 14
    mid_move = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    box_ratio = (vol / 100000000) / (high - low).replace(0, np.nan)
    d["EMV_RAW"] = (mid_move / box_ratio.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    d["EMV"] = d["EMV_RAW"].rolling(14).mean()

    # NVI / PVI
    pct_change = close.pct_change().fillna(0)
    nvi = [1000.0]
    pvi = [1000.0]
    for i in range(1, len(d)):
        if vol.iloc[i] < vol.iloc[i - 1]:
            nvi.append(nvi[-1] * (1 + pct_change.iloc[i]))
        else:
            nvi.append(nvi[-1])
        if vol.iloc[i] > vol.iloc[i - 1]:
            pvi.append(pvi[-1] * (1 + pct_change.iloc[i]))
        else:
            pvi.append(pvi[-1])
    d["NVI"] = pd.Series(nvi, index=d.index)
    d["PVI"] = pd.Series(pvi, index=d.index)
    d["NVI_SIGNAL"] = d["NVI"].ewm(span=20, adjust=False).mean()
    d["PVI_SIGNAL"] = d["PVI"].ewm(span=20, adjust=False).mean()

    # VA-OSC
    d["VA_OSC"] = d["ADI"].ewm(span=3, adjust=False).mean() - d["ADI"].ewm(span=10, adjust=False).mean()

    return d


# -----------------------------
# 警訊評分
# -----------------------------
@dataclass
class RiskResult:
    risk: int
    intraday_risk: int
    technical_risk: int
    recommendation: str
    warnings: List[str]
    severe_warnings: List[str]
    metrics: Dict[str, float]
    status_color: str


def add_warning(warnings: List[str], severe: List[str], text: str, points: int, is_severe: bool = False) -> int:
    msg = f"+{points}｜{text}"
    if is_severe:
        severe.append(msg)
    else:
        warnings.append(msg)
    return points


def risk_recommendation(score: int, severe_count: int, stop_loss_hit: bool = False) -> Tuple[str, str]:
    if stop_loss_hit:
        return "賣出警訊｜跌破停損價，優先處理", "#b91c1c"
    if severe_count >= 2 or score >= 86:
        return "重大警訊｜優先處理", "#b91c1c"
    if score >= 71:
        return "賣出警訊｜原買進理由可能失效", "#dc2626"
    if score >= 51:
        return "觀察／減碼｜警訊增加", "#f97316"
    if score >= 31:
        return "留意｜短線有轉弱跡象", "#ca8a04"
    return "續抱觀察｜條件尚未明顯破壞", "#15803d"


def evaluate_holding(row: pd.Series, hist_range: str = "6mo") -> Dict:
    code = str(row.get("code", "")).strip()
    raw_name = str(row.get("name", "")).strip()
    name = raw_name if raw_name and raw_name.lower() != "nan" else resolve_stock_name(code)
    buy_price = safe_float(row.get("buy_price"), 0)
    quantity = safe_float(row.get("quantity"), 0)
    stop_loss = safe_float(row.get("stop_loss"), 0)
    entry_score = safe_float(row.get("entry_score"), 0)

    df, meta = fetch_with_fallback(code, hist_range=hist_range)
    if df is None or len(df) < 35:
        return {
            "code": code, "name": name, "ticker": "", "buy_price": buy_price, "current": np.nan,
            "profit_pct": np.nan, "risk": 100, "intraday_risk": 0, "technical_risk": 0,
            "recommendation": "資料不足｜無法評估", "warnings": [meta.get("error", "抓取資料失敗")],
            "severe_warnings": [], "metrics": {}, "status_color": "#6b7280",
        }

    d = add_indicators(df)
    last = d.iloc[-1]
    prev = d.iloc[-2]

    # 最新行情值（日線 endpoint 盤中可能有延遲，盤後較完整）
    open_ = safe_float(last["open"])
    high = safe_float(last["high"])
    low = safe_float(last["low"])
    close = safe_float(last["close"])
    volume = safe_float(last["volume"], 0)
    prev_close = safe_float(prev["close"])
    prev_vol = safe_float(prev["volume"], 0)
    vol_ma5 = safe_float(last.get("vol_ma5"), np.nan)
    vol_ma20 = safe_float(last.get("vol_ma20"), np.nan)
    ma5 = safe_float(last.get("ma5"), np.nan)
    ma10 = safe_float(last.get("ma10"), np.nan)

    current = close
    profit_pct = ((current - buy_price) / buy_price * 100) if buy_price else np.nan
    open_premium = ((open_ - prev_close) / prev_close * 100) if prev_close else np.nan
    day_change_pct = ((current - prev_close) / prev_close * 100) if prev_close else np.nan
    drawdown_from_high = ((high - current) / high * 100) if high else np.nan
    upper_shadow_ratio = safe_float(last.get("upper_shadow_ratio"), 0)

    intraday_risk = 0
    technical_risk = 0
    warnings: List[str] = []
    severe: List[str] = []

    # 盤中量價警訊
    if buy_price and current < buy_price:
        intraday_risk += add_warning(warnings, severe, "現價跌破買進價", 10)
    if not math.isnan(open_) and current < open_:
        intraday_risk += add_warning(warnings, severe, "跌破今日開盤價，開盤強勢轉弱", 10)
    if current < prev_close:
        intraday_risk += add_warning(warnings, severe, "跌破昨收，盤中偏弱", 15)
    if not math.isnan(drawdown_from_high) and drawdown_from_high >= 3:
        intraday_risk += add_warning(warnings, severe, f"從當日高點回落 {drawdown_from_high:.2f}%", 10)
    if upper_shadow_ratio >= 0.45:
        intraday_risk += add_warning(warnings, severe, "形成長上影線，上方賣壓偏重", 10)
    if not math.isnan(vol_ma5) and volume > vol_ma5 * 1.8 and current < open_:
        intraday_risk += add_warning(warnings, severe, "爆量但價格壓回，疑似賣壓或出貨", 15, True)
    if not math.isnan(vol_ma5) and volume > vol_ma5 and current < open_:
        intraday_risk += add_warning(warnings, severe, "放量跌破開盤價", 20, True)
    if not math.isnan(open_premium) and open_premium > 5 and current < open_:
        intraday_risk += add_warning(warnings, severe, "開盤溢價過高後壓回，防追高失敗", 15)
    if stop_loss and current < stop_loss:
        intraday_risk += add_warning(warnings, severe, f"跌破停損價 {stop_loss:g}", 40, True)

    # 盤後/日線技術警訊
    if current < open_:
        technical_risk += add_warning(warnings, severe, "日K暫為黑K", 8)
    if upper_shadow_ratio >= 0.45:
        technical_risk += add_warning(warnings, severe, "日K長上影", 10)
    if current < open_ and not math.isnan(vol_ma5) and volume > vol_ma5 * 1.8:
        technical_risk += add_warning(warnings, severe, "爆量黑K，短線重大警訊", 25, True)
    if not math.isnan(ma5) and current < ma5:
        technical_risk += add_warning(warnings, severe, "跌破5日線", 15)
    if not math.isnan(ma10) and current < ma10:
        technical_risk += add_warning(warnings, severe, "跌破10日線", 20)
    if current < prev_close and not math.isnan(vol_ma5) and volume > vol_ma5:
        technical_risk += add_warning(warnings, severe, "放量下跌", 20, True)

    k = safe_float(last.get("K")); pk = safe_float(prev.get("K"))
    j = safe_float(last.get("J")); pj = safe_float(prev.get("J"))
    if k < pk and j < pj:
        technical_risk += add_warning(warnings, severe, "KD/KDJ 轉下，短線動能減弱", 10)

    mh = safe_float(last.get("MACD_hist")); pmh = safe_float(prev.get("MACD_hist"))
    if mh > 0 and mh < pmh:
        technical_risk += add_warning(warnings, severe, "MACD紅柱縮短，多方力道減弱", 10)
    if mh < 0 and mh < pmh:
        technical_risk += add_warning(warnings, severe, "MACD綠柱放大，空方力道增強", 20, True)

    rsi = safe_float(last.get("RSI"))
    if rsi < 50:
        technical_risk += add_warning(warnings, severe, "RSI跌破50，多方力道轉弱", 15)

    cci = safe_float(last.get("CCI"))
    if cci < 100:
        technical_risk += add_warning(warnings, severe, "CCI跌破或未站上+100，強勢條件減弱", 10)

    plus_di = safe_float(last.get("plus_DI")); minus_di = safe_float(last.get("minus_DI"))
    p_plus_di = safe_float(prev.get("plus_DI")); p_minus_di = safe_float(prev.get("minus_DI"))
    adx = safe_float(last.get("ADX")); p_adx = safe_float(prev.get("ADX"))
    if minus_di > p_minus_di and plus_di < p_plus_di:
        technical_risk += add_warning(warnings, severe, "-DI往上且+DI往下，空方轉強", 20, True)
    if minus_di > plus_di and adx > p_adx:
        technical_risk += add_warning(warnings, severe, "-DI大於+DI且ADX上升，下跌趨勢可能增強", 30, True)

    adi = safe_float(last.get("ADI")); padi = safe_float(prev.get("ADI"))
    va = safe_float(last.get("VA_OSC")); pva = safe_float(prev.get("VA_OSC"))
    if current > prev_close and adi < padi:
        technical_risk += add_warning(warnings, severe, "股價漲但A/DI下降，資金流向背離", 15)
    if pva > 0 and va < 0:
        technical_risk += add_warning(warnings, severe, "VA-OSC由正轉負，資金量能動能轉弱", 20, True)
    elif current > prev_close and va < pva:
        technical_risk += add_warning(warnings, severe, "股價漲但VA-OSC下降，量能動能背離", 10)

    pct_b = safe_float(last.get("PCT_B"))
    if pct_b < 0.5:
        technical_risk += add_warning(warnings, severe, "%B跌破0.5，股價回到布林中軌下方", 10)

    mtm = safe_float(last.get("MTM")); pmtm = safe_float(prev.get("MTM"))
    roc = safe_float(last.get("ROC")); proc = safe_float(prev.get("ROC"))
    if (pmtm > 0 and mtm < 0) or (proc > 0 and roc < 0):
        technical_risk += add_warning(warnings, severe, "MTM/ROC由正轉負，動能轉弱", 15)

    pvi = safe_float(last.get("PVI")); ppvi = safe_float(prev.get("PVI"))
    nvi = safe_float(last.get("NVI")); pnvi = safe_float(prev.get("NVI"))
    if pvi < ppvi:
        technical_risk += add_warning(warnings, severe, "PVI下滑，放量人氣轉弱", 8)
    if nvi < pnvi:
        technical_risk += add_warning(warnings, severe, "NVI下滑，縮量支撐轉弱", 8)

    br = safe_float(last.get("BR"))
    vr = safe_float(last.get("VR"))
    emv = safe_float(last.get("EMV")); pemv = safe_float(prev.get("EMV"))
    if br > 300 and upper_shadow_ratio >= 0.45:
        technical_risk += add_warning(warnings, severe, "BR過熱且長上影，追價買盤可能退潮", 15)
    if vr > 350 and upper_shadow_ratio >= 0.45:
        technical_risk += add_warning(warnings, severe, "VR過熱且長上影，成交量人氣可能過熱轉弱", 15)
    if emv < pemv and current >= prev_close:
        technical_risk += add_warning(warnings, severe, "股價未跌但EMV下降，上攻效率變差", 10)

    # 買進理由失效加權：買進時分數高，但現在警訊多，要提高風險提示
    if entry_score >= 20 and technical_risk >= 50:
        technical_risk += add_warning(warnings, severe, "買進時屬高分標的，但目前技術警訊已明顯增加", 10)

    total_risk = int(min(100, max(0, intraday_risk + technical_risk)))
    stop_loss_hit = bool(stop_loss and current < stop_loss)
    recommendation, color = risk_recommendation(total_risk, len(severe), stop_loss_hit)

    metrics = {
        "open": open_, "high": high, "low": low, "prev_close": prev_close,
        "volume": volume, "vol_ma5": vol_ma5, "vol_ma20": vol_ma20,
        "profit_pct": profit_pct, "open_premium": open_premium, "day_change_pct": day_change_pct,
        "drawdown_from_high": drawdown_from_high, "RSI": rsi, "CCI": cci, "K": k, "J": j,
        "MACD_hist": mh, "plus_DI": plus_di, "minus_DI": minus_di, "ADX": adx,
        "PCT_B": pct_b, "MTM": mtm, "ROC": roc, "BR": br, "VR": vr, "EMV": emv,
        "ADI": adi, "VA_OSC": va, "NVI": nvi, "PVI": pvi,
    }

    return {
        "code": code.split(".")[0],
        "name": name,
        "ticker": meta.get("resolved_ticker", ""),
        "buy_price": buy_price,
        "quantity": quantity,
        "stop_loss": stop_loss,
        "current": current,
        "profit_pct": profit_pct,
        "risk": total_risk,
        "intraday_risk": int(min(100, intraday_risk)),
        "technical_risk": int(min(100, technical_risk)),
        "recommendation": recommendation,
        "warnings": warnings,
        "severe_warnings": severe,
        "metrics": metrics,
        "status_color": color,
        "latest_date": str(pd.to_datetime(last["datetime"]).date()) if "datetime" in last else "",
    }


# -----------------------------
# UI
# -----------------------------
def css():
    st.markdown(
        """
        <style>
        .big-title {font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem;}
        .subtle {color:#6b7280; font-size:0.95rem;}
        .risk-card {border:1px solid #e5e7eb; border-radius:14px; padding:14px; margin:10px 0; background:#ffffff;}
        .risk-pill {display:inline-block; color:white; padding:4px 10px; border-radius:999px; font-weight:700;}
        .small-note {font-size:0.9rem; color:#4b5563;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_empty_holdings() -> pd.DataFrame:
    return pd.DataFrame(columns=DEFAULT_HOLDING_COLUMNS)




def holdings_to_records_json(df: pd.DataFrame) -> str:
    """把持股資料轉成可存入瀏覽器 localStorage 的 JSON 字串。"""
    h = normalize_holdings_df(df)
    # 只保留固定欄位，避免 data_editor 產生額外欄位造成版本不相容
    records = h[DEFAULT_HOLDING_COLUMNS].to_dict(orient="records")
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def holdings_from_records_json(raw: str) -> pd.DataFrame:
    """從瀏覽器 localStorage/備份檔的 JSON 還原持股資料。"""
    if not raw:
        return make_empty_holdings()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "holdings" in data:
            data = data["holdings"]
        if not isinstance(data, list):
            return make_empty_holdings()
        return normalize_holdings_df(pd.DataFrame(data))
    except Exception:
        return make_empty_holdings()


def storage_available() -> bool:
    return streamlit_js_eval is not None


def load_holdings_from_browser_once():
    """第一次進入時，從目前瀏覽器 localStorage 載入持股資料。"""
    if st.session_state.get("browser_storage_loaded", False):
        return
    if not storage_available():
        st.session_state.browser_storage_loaded = True
        st.session_state.browser_storage_status = "本機儲存元件未載入，暫時只使用本次頁面資料。"
        return

    raw = streamlit_js_eval(
        js_expressions=f"localStorage.getItem('{BROWSER_STORAGE_KEY}') || '__EMPTY__'",
        key="load_browser_holdings_v5",
    )
    if raw is None:
        st.info("正在讀取瀏覽器本機儲存資料，若畫面未自動更新，請稍等幾秒後按一下重新整理。")
        return

    st.session_state.browser_storage_loaded = True
    if raw != "__EMPTY__":
        restored = holdings_from_records_json(raw)
        if not restored.empty:
            st.session_state.holdings_df = restored
            st.session_state.browser_storage_status = f"已從瀏覽器本機儲存載入 {len(restored)} 筆持股資料。"
            st.rerun()
        else:
            st.session_state.browser_storage_status = "瀏覽器中有舊資料，但格式無法還原，已略過。"
    else:
        st.session_state.browser_storage_status = "此瀏覽器尚無已儲存持股資料。"


def save_holdings_to_browser():
    """將目前持股資料自動寫入瀏覽器 localStorage。"""
    if not st.session_state.get("browser_storage_loaded", False):
        return
    if not storage_available():
        return
    payload = holdings_to_records_json(st.session_state.get("holdings_df", make_empty_holdings()))
    checksum = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
    # 用不同 key 讓 Streamlit 在資料變動後確實執行新的 JS；不改變畫面高度
    streamlit_js_eval(
        js_expressions=f"localStorage.setItem('{BROWSER_STORAGE_KEY}', {json.dumps(payload, ensure_ascii=False)}); true;",
        key=f"save_browser_holdings_{checksum}",
    )


def clear_browser_storage_js():
    """清除目前瀏覽器 localStorage 中的持股資料。"""
    if not storage_available():
        return
    streamlit_js_eval(
        js_expressions=f"localStorage.removeItem('{BROWSER_STORAGE_KEY}'); true;",
        key=f"clear_browser_holdings_{date.today().isoformat()}_{hashlib.md5(str(pd.Timestamp.now()).encode()).hexdigest()[:8]}",
    )


def backup_restore_panel():
    st.subheader("資料備份 / 還原")
    st.caption("持股會自動存到目前瀏覽器；但若換手機、換瀏覽器或清除 Safari 網站資料，仍可能消失。建議偶爾下載備份。")
    current_json = holdings_to_records_json(st.session_state.get("holdings_df", make_empty_holdings()))
    st.download_button(
        "下載持股備份 JSON",
        data=current_json.encode("utf-8"),
        file_name="台股持股警訊監控器_持股備份.json",
        mime="application/json",
        use_container_width=True,
    )
    uploaded = st.file_uploader("匯入持股備份 JSON", type=["json"], key="restore_json_upload")
    if uploaded is not None:
        if st.button("確認匯入備份並覆蓋目前資料", type="primary"):
            try:
                raw = uploaded.read().decode("utf-8")
                restored = holdings_from_records_json(raw)
                st.session_state.holdings_df = restored
                st.session_state.monitor_results = []
                st.session_state.browser_storage_loaded = True
                st.success(f"已匯入 {len(restored)} 筆資料，並會自動存入此瀏覽器。")
                st.rerun()
            except Exception as e:
                st.error(f"匯入失敗：{e}")

def init_holdings_state():
    if "holdings_df" not in st.session_state:
        st.session_state.holdings_df = make_empty_holdings()
    if "monitor_stage" not in st.session_state:
        st.session_state.monitor_stage = "input"
    if "browser_storage_loaded" not in st.session_state:
        st.session_state.browser_storage_loaded = False
    if "browser_storage_status" not in st.session_state:
        st.session_state.browser_storage_status = "尚未讀取瀏覽器本機儲存。"


def active_holdings_df(df: pd.DataFrame) -> pd.DataFrame:
    h = normalize_holdings_df(df)
    if h.empty:
        return h
    return h[h["status"].astype(str).str.strip().ne("已賣出")].reset_index(drop=True)


def sold_holdings_df(df: pd.DataFrame) -> pd.DataFrame:
    h = normalize_holdings_df(df)
    if h.empty:
        return h
    return h[h["status"].astype(str).str.strip().eq("已賣出")].reset_index(drop=True)


def holding_label(row: pd.Series, idx: int) -> str:
    code = str(row.get("code", "")).strip()
    name = str(row.get("name", "")).strip()
    bp = safe_float(row.get("buy_price"), 0)
    return f"{idx}｜{code} {name}｜買進 {bp:g}"


def delete_holding_by_index(idx: int):
    h = normalize_holdings_df(st.session_state.holdings_df)
    if 0 <= idx < len(h):
        st.session_state.holdings_df = h.drop(index=idx).reset_index(drop=True)
        st.session_state.monitor_results = []


def mark_holding_sold(idx: int, sell_date: date, sell_price: float, sell_reason: str):
    h = normalize_holdings_df(st.session_state.holdings_df)
    if 0 <= idx < len(h):
        h.at[idx, "status"] = "已賣出"
        h.at[idx, "sell_date"] = str(sell_date)
        h.at[idx, "sell_price"] = float(sell_price or 0)
        h.at[idx, "sell_reason"] = sell_reason or "已賣出，停止監控"
        st.session_state.holdings_df = h.reset_index(drop=True)
        st.session_state.monitor_results = []


def append_holding(new_row: Dict):
    init_holdings_state()
    base = normalize_holdings_df(st.session_state.holdings_df)
    added = normalize_holdings_df(pd.DataFrame([new_row]))
    if not added.empty:
        st.session_state.holdings_df = normalize_holdings_df(pd.concat([base, added], ignore_index=True))


def manual_add_form():
    st.subheader("新增持股")
    st.caption("只要輸入股票代號，股票名稱會自動帶出；完成後再進入監控器。")

    c1, c2 = st.columns(2)
    code = c1.text_input("股票代號 *", placeholder="例如 1718", key="manual_code")
    code_clean = clean_code(code)
    auto_name = resolve_stock_name(code_clean) if code_clean else ""
    c2.markdown("**股票名稱（自動帶出）**")
    if code_clean and auto_name:
        c2.success(auto_name)
    elif code_clean:
        c2.warning("暫時查不到名稱，加入後仍可監控；也可稍後按『自動補齊名稱』。")
    else:
        c2.info("輸入股票代號後自動帶出")

    c3, c4 = st.columns(2)
    buy_date = c3.date_input("買進日期", value=date.today(), key="manual_buy_date")
    buy_price = c4.number_input("買進價 *", min_value=0.0, value=0.0, step=0.05, format="%.2f", key="manual_buy_price")

    c5, c6, c7 = st.columns(3)
    quantity = c5.number_input("股數", min_value=0.0, value=1000.0, step=100.0, key="manual_quantity")
    stop_loss = c6.number_input("停損價", min_value=0.0, value=0.0, step=0.05, format="%.2f", key="manual_stop_loss")
    entry_score = c7.number_input("買進時掃描器分數", min_value=0.0, value=0.0, step=1.0, key="manual_entry_score")

    entry_reasons = st.text_area("買進理由", placeholder="例如：選股掃描器高分、量增放量、KD向上、MACD翻紅", key="manual_entry_reasons")
    note = st.text_input("備註", placeholder="可空白", key="manual_note")

    submitted = st.button("加入持股清單", type="primary")
    if submitted:
        if not code_clean:
            st.error("請至少輸入股票代號。")
        elif buy_price <= 0:
            st.error("請輸入買進價。")
        else:
            name = auto_name or resolve_stock_name(code_clean)
            append_holding({
                "code": code_clean,
                "name": name,
                "buy_date": str(buy_date),
                "buy_price": buy_price,
                "quantity": quantity,
                "stop_loss": stop_loss,
                "entry_score": entry_score,
                "entry_reasons": entry_reasons,
                "note": note,
                "status": "監控中",
                "sell_date": "",
                "sell_price": 0,
                "sell_reason": "",
            })
            st.success(f"已加入 {code_clean} {name}")



def holdings_editor():
    init_holdings_state()
    st.subheader("目前持股清單")
    st.caption("可以直接在表格中修改；若已賣出建議用下方『標記已賣出』保留紀錄，輸入錯誤再用『刪除持股』。")
    edited = st.data_editor(
        normalize_holdings_df(st.session_state.holdings_df),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "code": st.column_config.TextColumn("股票代號", help="例如 1718、2330", required=True),
            "name": st.column_config.TextColumn("股票名稱"),
            "buy_date": st.column_config.TextColumn("買進日期"),
            "buy_price": st.column_config.NumberColumn("買進價", min_value=0.0, step=0.05, required=True),
            "quantity": st.column_config.NumberColumn("股數", min_value=0.0, step=100.0),
            "stop_loss": st.column_config.NumberColumn("停損價", min_value=0.0, step=0.05),
            "entry_score": st.column_config.NumberColumn("買進時分數", min_value=0.0, step=1.0),
            "entry_reasons": st.column_config.TextColumn("買進理由"),
            "note": st.column_config.TextColumn("備註"),
            "status": st.column_config.SelectboxColumn("狀態", options=["監控中", "已賣出"], required=True),
            "sell_date": st.column_config.TextColumn("賣出日期"),
            "sell_price": st.column_config.NumberColumn("賣出價", min_value=0.0, step=0.05),
            "sell_reason": st.column_config.TextColumn("賣出原因"),
        },
        key="holdings_editor_v4",
    )
    st.session_state.holdings_df = normalize_holdings_df(edited)
    return st.session_state.holdings_df


def holding_action_panel():
    h = normalize_holdings_df(st.session_state.holdings_df)
    if h.empty:
        return
    st.subheader("持股操作")
    st.caption("已經賣掉的股票，建議選『標記已賣出』，會保留交易紀錄並停止監控；輸入錯誤才用『刪除持股』。")

    labels = [holding_label(row, i) for i, row in h.iterrows()]
    selected_label = st.selectbox("選擇要處理的股票", labels, key="holding_action_select")
    selected_idx = int(selected_label.split("｜", 1)[0])
    selected_row = h.iloc[selected_idx]
    selected_code = selected_row.get("code", "")
    selected_name = selected_row.get("name", "")

    action = st.radio("動作", ["標記已賣出／停止監控", "刪除持股（不保留紀錄）"], horizontal=True, key="holding_action_radio")

    if action.startswith("標記"):
        c1, c2 = st.columns(2)
        sell_date = c1.date_input("賣出日期", value=date.today(), key="sold_date_input")
        default_sell = safe_float(selected_row.get("sell_price"), 0)
        if default_sell <= 0:
            default_sell = safe_float(selected_row.get("buy_price"), 0)
        sell_price = c2.number_input("賣出價", min_value=0.0, value=float(default_sell), step=0.05, format="%.2f", key="sold_price_input")
        sell_reason = st.text_input("賣出原因", placeholder="例如：達停利、跌破5日線、爆量黑K、手動出場", key="sold_reason_input")
        if st.button(f"確認標記已賣出：{selected_code} {selected_name}", type="primary"):
            if sell_price <= 0:
                st.error("請輸入賣出價，方便保留損益紀錄。")
            else:
                mark_holding_sold(selected_idx, sell_date, sell_price, sell_reason)
                st.success(f"已將 {selected_code} {selected_name} 標記為已賣出，並停止監控。")
                st.rerun()
    else:
        st.warning("刪除後不保留這筆持股紀錄。若是正常賣出，建議改用『標記已賣出』。")
        confirm = st.checkbox(f"我確認要刪除 {selected_code} {selected_name}", key="delete_confirm")
        if st.button("確認刪除持股", type="primary", disabled=not confirm):
            delete_holding_by_index(selected_idx)
            st.success(f"已刪除 {selected_code} {selected_name}。")
            st.rerun()


def sold_records_panel():
    sold = sold_holdings_df(st.session_state.holdings_df)
    st.subheader("已賣出紀錄")
    if sold.empty:
        st.caption("目前沒有已賣出紀錄。")
        return
    out = sold.copy()
    out["realized_profit_pct"] = np.where(
        out["buy_price"] > 0,
        (out["sell_price"] - out["buy_price"]) / out["buy_price"] * 100,
        np.nan,
    )
    view_cols = ["code", "name", "buy_date", "buy_price", "sell_date", "sell_price", "realized_profit_pct", "quantity", "sell_reason"]
    st.dataframe(
        out[view_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "code": "股票代號",
            "name": "股票名稱",
            "buy_date": "買進日期",
            "buy_price": st.column_config.NumberColumn("買進價", format="%.2f"),
            "sell_date": "賣出日期",
            "sell_price": st.column_config.NumberColumn("賣出價", format="%.2f"),
            "realized_profit_pct": st.column_config.NumberColumn("實現損益%", format="%.2f%%"),
            "quantity": "股數",
            "sell_reason": "賣出原因",
        },
    )
    st.download_button(
        "下載已賣出紀錄 CSV",
        data=out.to_csv(index=False).encode("utf-8-sig"),
        file_name="已賣出紀錄.csv",
        mime="text/csv",
    )

def holdings_input_page():
    init_holdings_state()
    st.info("第 1 步：先新增或編輯持股資料。完成後按『儲存並進入監控器』。")
    manual_add_form()
    st.divider()
    holdings = holdings_editor()
    st.divider()
    holding_action_panel()
    st.divider()
    sold_records_panel()
    st.divider()
    backup_restore_panel()

    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 2])
    if c1.button("清空持股資料"):
        st.session_state.holdings_df = make_empty_holdings()
        st.session_state.monitor_results = []
        st.session_state.browser_storage_loaded = True
        st.rerun()
    if c2.button("載入一筆範例"):
        append_holding({
            "code": "1718", "name": "中纖", "buy_date": str(date.today()), "buy_price": 12.50,
            "quantity": 1000, "stop_loss": 11.80, "entry_score": 28,
            "entry_reasons": "選股掃描器分數高；量增放量；KD向上", "note": "範例資料可刪除",
            "status": "監控中", "sell_date": "", "sell_price": 0, "sell_reason": ""
        })
        st.rerun()
    if c3.button("自動補齊名稱"):
        st.session_state.holdings_df = autofill_holding_names(st.session_state.holdings_df)
        st.success("已依股票代號補齊可查到的中文名稱。")
        st.rerun()
    go = c4.button("儲存並進入監控器", type="primary", use_container_width=True)
    if go:
        holdings = autofill_holding_names(st.session_state.holdings_df)
        active = active_holdings_df(holdings)
        if active.empty:
            st.error("目前沒有監控中的持股。請新增持股，或把狀態改成『監控中』。")
        elif (active["buy_price"] <= 0).any():
            st.error("所有監控中持股都要輸入買進價，才能計算損益與風險。")
        else:
            st.session_state.holdings_df = holdings
            st.session_state.monitor_stage = "monitor"
            st.rerun()


def monitor_page(hist_range: str, auto_refresh: bool):
    init_holdings_state()
    all_holdings = normalize_holdings_df(st.session_state.holdings_df)
    holdings = active_holdings_df(all_holdings)

    top1, top2 = st.columns([1, 3])
    if top1.button("← 返回持股輸入"):
        st.session_state.monitor_stage = "input"
        st.rerun()
    top2.caption("第 2 步：進入監控器後，按『開始檢查持股警訊』取得盤中／盤後警訊與建議動作。")

    if holdings.empty:
        st.warning("目前沒有監控中的持股，請返回持股輸入。")
        return

    st.subheader("監控中持股")
    st.caption("狀態為『已賣出』的股票不會進入警訊監控。")
    st.dataframe(holdings, use_container_width=True, hide_index=True)
    sold = sold_holdings_df(all_holdings)
    if not sold.empty:
        with st.expander(f"已賣出／停止監控紀錄：{len(sold)} 檔"):
            sold_records_panel()

    run = st.button("開始檢查持股警訊", type="primary")
    if auto_refresh:
        run = True
        try:
            st.autorefresh(interval=60 * 1000, key="refresh")
        except Exception:
            pass

    if run:
        results = []
        progress = st.progress(0)
        for i, (_, row) in enumerate(holdings.iterrows(), start=1):
            with st.spinner(f"檢查 {row.get('code')} 中..."):
                results.append(evaluate_holding(row, hist_range=hist_range))
            progress.progress(i / max(len(holdings), 1))
        st.session_state.monitor_results = results

    results = st.session_state.get("monitor_results", [])
    if results:
        summary = build_summary_df(results)
        st.subheader("持股總覽")
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "損益%": st.column_config.NumberColumn(format="%.2f%%"),
                "買進價": st.column_config.NumberColumn(format="%.2f"),
                "現價": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.download_button(
            "下載本次警訊結果 CSV",
            data=summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="持股警訊監控結果.csv",
            mime="text/csv",
        )
        st.subheader("單檔詳細警訊")
        render_result_cards(results)


def build_summary_df(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "股票代號": r.get("code"),
            "股票名稱": r.get("name"),
            "買進價": r.get("buy_price"),
            "現價": r.get("current"),
            "損益%": r.get("profit_pct"),
            "盤中風險": r.get("intraday_risk"),
            "技術風險": r.get("technical_risk"),
            "總風險": r.get("risk"),
            "建議動作": r.get("recommendation"),
            "重大警訊數": len(r.get("severe_warnings", [])),
            "主要警訊": "；".join((r.get("severe_warnings", []) + r.get("warnings", []))[:3]),
            "資料日期": r.get("latest_date", ""),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["總風險", "重大警訊數"], ascending=[False, False]).reset_index(drop=True)
    return out


def render_result_cards(results: List[Dict]):
    for r in sorted(results, key=lambda x: x.get("risk", 0), reverse=True):
        color = r.get("status_color", "#6b7280")
        title = f"{r.get('code')} {r.get('name', '')}"
        st.markdown(
            f"""
            <div class="risk-card">
              <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;">
                <div style="font-size:1.25rem; font-weight:800;">{title}</div>
                <div class="risk-pill" style="background:{color};">風險 {r.get('risk', 0)}｜{r.get('recommendation')}</div>
              </div>
              <div class="small-note">買進價 {fmt_price(r.get('buy_price'))}｜現價 {fmt_price(r.get('current'))}｜損益 {fmt_pct(r.get('profit_pct'))}｜盤中風險 {r.get('intraday_risk')}｜技術風險 {r.get('technical_risk')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"查看 {title} 詳細警訊與指標"):
            col1, col2, col3, col4 = st.columns(4)
            m = r.get("metrics", {})
            col1.metric("現價", fmt_price(r.get("current")), fmt_pct(m.get("day_change_pct")))
            col2.metric("損益", fmt_pct(r.get("profit_pct")))
            col3.metric("開盤溢價率", fmt_pct(m.get("open_premium")))
            col4.metric("從高點回落", fmt_pct(-safe_float(m.get("drawdown_from_high"))))

            if r.get("severe_warnings"):
                st.error("重大警訊：\n" + "\n".join([f"- {x}" for x in r["severe_warnings"]]))
            if r.get("warnings"):
                st.warning("一般警訊：\n" + "\n".join([f"- {x}" for x in r["warnings"]]))
            if not r.get("warnings") and not r.get("severe_warnings"):
                st.success("目前未偵測到明顯警訊。")

            metrics_df = pd.DataFrame([
                {"項目": "開盤", "數值": fmt_price(m.get("open"))},
                {"項目": "最高", "數值": fmt_price(m.get("high"))},
                {"項目": "最低", "數值": fmt_price(m.get("low"))},
                {"項目": "昨收", "數值": fmt_price(m.get("prev_close"))},
                {"項目": "成交量", "數值": f"{safe_float(m.get('volume'), 0):,.0f}"},
                {"項目": "RSI", "數值": fmt_price(m.get("RSI"))},
                {"項目": "CCI", "數值": fmt_price(m.get("CCI"))},
                {"項目": "K / J", "數值": f"{fmt_price(m.get('K'))} / {fmt_price(m.get('J'))}"},
                {"項目": "+DI / -DI / ADX", "數值": f"{fmt_price(m.get('plus_DI'))} / {fmt_price(m.get('minus_DI'))} / {fmt_price(m.get('ADX'))}"},
                {"項目": "MACD柱", "數值": fmt_price(m.get("MACD_hist"))},
                {"項目": "%B", "數值": fmt_price(m.get("PCT_B"))},
                {"項目": "MTM / ROC", "數值": f"{fmt_price(m.get('MTM'))} / {fmt_price(m.get('ROC'))}"},
                {"項目": "BR / VR", "數值": f"{fmt_price(m.get('BR'))} / {fmt_price(m.get('VR'))}"},
                {"項目": "EMV", "數值": fmt_price(m.get("EMV"))},
                {"項目": "VA-OSC", "數值": fmt_price(m.get("VA_OSC"))},
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def render_rules():
    st.subheader("v5 風險分數、持股管理與本機儲存規則")
    st.markdown(
        """
這套工具跟前面的選股掃描器相反：**風險分數越高越危險**。

v5 新增：持股資料會自動儲存在目前瀏覽器的本機儲存空間（localStorage）。同一支手機、同一個瀏覽器再次打開同一網址時，會自動載入上次輸入的持股。

| 風險分數 | 判斷 |
|---:|---|
| 0～30 | 續抱觀察：條件尚未明顯破壞 |
| 31～50 | 留意：短線有轉弱跡象 |
| 51～70 | 觀察／減碼：警訊增加 |
| 71～85 | 賣出警訊：原買進理由可能失效 |
| 86～100 | 重大警訊：優先處理 |

主要警訊包含：跌破買進價、跌破開盤價、跌破昨收、從高點回落、長上影、爆量壓回、跌破停損價、爆量黑K、跌破5/10日線、放量下跌、KD/KDJ轉下、MACD轉弱、RSI跌破50、CCI跌破+100、-DI轉強、A/DI或VA-OSC背離、%B跌破0.5、MTM/ROC轉負、NVI/PVI下滑等。

**持股管理建議：**正常賣出請使用「標記已賣出／停止監控」，系統會保留買進價、賣出價、賣出原因與實現損益；只有輸入錯誤或完全不想保留紀錄時，才使用「刪除持股」。
        """
    )


def main():
    css()
    init_holdings_state()
    st.markdown('<div class="big-title">🚨 台股持股警訊監控器 v5</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">買進後監控盤中或盤後重大警訊。v5 新增瀏覽器本機儲存，重新打開同一網址時會自動帶回持股資料。</div>', unsafe_allow_html=True)
    load_holdings_from_browser_once()
    if st.session_state.get("browser_storage_status"):
        st.caption(st.session_state.browser_storage_status)

    with st.sidebar:
        st.header("設定")
        hist_range = st.selectbox("歷史資料區間", ["3mo", "6mo", "1y"], index=1)
        auto_refresh = st.checkbox("盤中自動更新（約60秒）", value=False)
        st.caption("雲端版資料可能有延遲；實際下單前請以券商APP確認。")
        st.caption("持股資料儲存在目前瀏覽器；不同手機或不同瀏覽器不會同步。")
        if st.button("清除行情快取"):
            st.cache_data.clear()
            st.success("已清除快取，請重新檢查。")

        st.divider()
        st.write("目前頁面：", "持股輸入" if st.session_state.monitor_stage == "input" else "監控器")
        if st.button("切換到持股輸入"):
            st.session_state.monitor_stage = "input"
            st.rerun()
        if st.button("切換到監控器"):
            st.session_state.monitor_stage = "monitor"
            st.rerun()

    tab1, tab2 = st.tabs(["持股輸入 / 監控", "評分規則"])

    with tab1:
        if st.session_state.monitor_stage == "input":
            holdings_input_page()
        else:
            monitor_page(hist_range, auto_refresh)

    with tab2:
        render_rules()

    save_holdings_to_browser()
    st.caption("提醒：此工具依公開行情與技術指標計算，資料可能延遲或中斷；請勿把結果當成唯一買賣依據。")

if __name__ == "__main__":
    main()
