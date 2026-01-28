from __future__ import annotations

import os
import argparse
import datetime as dt
import json
import logging
import random
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import akshare as ak
import efinance as ef
import pandas as pd
import tushare as ts
from mootdx.quotes import Quotes
from mootdx.reader import Reader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fetch_mktcap")

# 屏蔽第三方库多余 INFO 日志
for noisy in ("httpx", "urllib3", "_client", "akshare"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------- 市值快照 --------------------------- #

def _get_mktcap_efinance(out_dir: Path) -> pd.DataFrame:
    """使用 efinance 获取股票列表（不包含市值）"""
    # 首先尝试使用指定的输出目录中的本地数据
    if out_dir.exists():
        local_codes = [p.stem for p in out_dir.glob("*.csv")]
        if local_codes:
            df = pd.DataFrame({"code": local_codes, "mktcap": 0.0})
            logger.info("从数据目录 %s 读取到 %d 只股票", out_dir, len(local_codes))
            return df

    # 其次使用 appendix.json
    try:
        with open("appendix.json", "r", encoding="utf-8") as f:
            appendix_codes = json.load(f)["data"]
        df = pd.DataFrame({"code": appendix_codes, "mktcap": 0.0})
        logger.info("从 appendix.json 读取到 %d 只股票", len(appendix_codes))
        return df
    except FileNotFoundError:
        pass

    # 再次检查当前目录的 data 子目录
    data_dir = Path("data")
    if data_dir.exists():
        local_codes = [p.stem for p in data_dir.glob("*.csv")]
        if local_codes:
            df = pd.DataFrame({"code": local_codes, "mktcap": 0.0})
            logger.info("从 ./data 目录读取到 %d 只股票", len(local_codes))
            return df

    # 最后从 efinance 获取股票列表
    for attempt in range(1, 4):
        try:
            df = ef.stock.get_stock_list()
            break
        except Exception as e:
            logger.warning("Efinance 获取股票列表失败(%d/3): %s", attempt, e)
            time.sleep(random.uniform(1, 3) * attempt)
    else:
        raise RuntimeError("Efinance 连续三次拉取股票列表失败！")

    df = df[["代码"]].rename(columns={"代码": "code"})
    df["mktcap"] = 0.0
    logger.info("从 efinance 获取到 %d 只股票（无市值信息）", len(df))
    return df


def _get_mktcap_ak() -> pd.DataFrame:
    """实时快照，返回列：code, mktcap（单位：元）"""
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            logger.warning("AKShare 获取市值快照失败(%d/3): %s", attempt, e)
            time.sleep(random.uniform(1, 3) * attempt)
    else:
        raise RuntimeError("AKShare 连续三次拉取市值快照失败！")

    df = df[["代码", "总市值"]].rename(columns={"代码": "code", "总市值": "mktcap"})
    df["mktcap"] = pd.to_numeric(df["mktcap"], errors="coerce")
    return df


# --------------------------- 股票池筛选 --------------------------- #

def get_constituents(
    min_cap: float,
    max_cap: float,
    small_player: bool,
    mktcap_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    df = mktcap_df if mktcap_df is not None else _get_mktcap_ak()

    cond = (df["mktcap"] >= min_cap) & (df["mktcap"] <= max_cap)
    if small_player:
        cond &= ~df["code"].str.startswith(("300", "301", "688", "8", "4"))

    codes = df.loc[cond, "code"].str.zfill(6).tolist()

    # 附加股票池 appendix.json
    try:
        with open("appendix.json", "r", encoding="utf-8") as f:
            appendix_codes = json.load(f)["data"]
    except FileNotFoundError:
        appendix_codes = []
    codes = list(dict.fromkeys(appendix_codes + codes))  # 去重保持顺序

    logger.info("筛选得到 %d 只股票", len(codes))
    return codes


# --------------------------- 历史 K 线抓取 --------------------------- #
COLUMN_MAP_HIST_AK = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

_FREQ_MAP = {
    0: "5m",
    1: "15m",
    2: "30m",
    3: "1h",
    4: "day",
    5: "week",
    6: "mon",
    7: "1m",
    8: "1m",
    9: "day",
    10: "3mon",
    11: "year",
}


# ---------- Tushare 工具函数 ---------- #

def _to_ts_code(code: str) -> str:
    return f"{code.zfill(6)}.SH" if code.startswith(("60", "68", "9")) else f"{code.zfill(6)}.SZ"


def _get_kline_tushare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    adj_flag = None if adjust == "" else adjust
    for attempt in range(1, 4):
        try:
            df = ts.pro_bar(
                ts_code=ts_code,
                adj=adj_flag,
                start_date=start,
                end_date=end,
                freq="D",
            )
            break
        except Exception as e:
            logger.warning("Tushare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return df.sort_values("date").reset_index(drop=True)


# ---------- AKShare 工具函数 ---------- #

def _get_kline_akshare(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            )
            break
        except Exception as e:
            logger.warning("AKShare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = (
        df[list(COLUMN_MAP_HIST_AK)]
        .rename(columns=COLUMN_MAP_HIST_AK)
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df[["date", "open", "close", "high", "low", "volume"]]
    return df.sort_values("date").reset_index(drop=True)


# ---------- Mootdx 工具函数 ---------- #

_mootdx_local = threading.local()

def _get_kline_mootdx(
    code: str, start: str, end: str, adjust: str, freq_code: int, tdx_dir: Optional[str]
) -> pd.DataFrame:
    symbol = code.zfill(6)
    freq = _FREQ_MAP.get(freq_code, "day")

    try:
        if tdx_dir:
            reader = Reader.factory(market="std", tdxdir=tdx_dir)
            df = reader.daily(symbol=symbol)
        else:
            if not hasattr(_mootdx_local, "client"):
                # 初始化线程独立的 client
                _mootdx_local.client = Quotes.factory(market="std")
            
            client = _mootdx_local.client
            try:
                df = client.bars(symbol=symbol, frequency=freq, adjust=adjust or None)
            except Exception:
                # 尝试重连
                logger.warning("Mootdx 连接异常，尝试重连...")
                _mootdx_local.client = Quotes.factory(market="std")
                client = _mootdx_local.client
                df = client.bars(symbol=symbol, frequency=freq, adjust=adjust or None)

    except Exception as e:
        logger.warning("Mootdx 拉取 %s 失败: %s", code, e)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={"datetime": "date", "vol": "volume"}
    )
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    df = df[(df["date"].dt.date >= start_ts.date()) & (df["date"].dt.date <= end_ts.date())].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "close", "high", "low", "volume"]]


# ---------- Efinance 工具函数 ---------- #

def _get_kline_efinance(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            df = ef.stock.get_quote_history(code)
            break
        except Exception as e:
            logger.warning("Efinance 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(min(8, 2 * attempt))
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = (
        df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )

    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    df = df[(df["date"].dt.date >= start_ts.date()) & (df["date"].dt.date <= end_ts.date())].copy()

    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )

    df = df[["date", "open", "close", "high", "low", "volume"]].sort_values("date").reset_index(drop=True)
    return df


# ---------- 通用接口 ---------- #

def get_kline(
    code: str,
    start: str,
    end: str,
    adjust: str,
    datasource: str,
    freq_code: int = 4,
    tdx_dir: Optional[str] = None,
) -> pd.DataFrame:
    if datasource == "tushare":
        return _get_kline_tushare(code, start, end, adjust)
    elif datasource == "akshare":
        return _get_kline_akshare(code, start, end, adjust)
    elif datasource == "mootdx":
        return _get_kline_mootdx(code, start, end, adjust, freq_code, tdx_dir)
    elif datasource == "efinance":
        return _get_kline_efinance(code, start, end, adjust)
    else:
        raise ValueError("datasource 仅支持 'tushare', 'akshare', 'mootdx' 或 'efinance'")


# ---------- 数据校验 ---------- #

def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


def drop_dup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]


# ---------- 单只股票抓取 ---------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    incremental: bool,
    datasource: str,
    freq_code: int,
    tdx_dir: Optional[str] = None,
):
    csv_path = out_dir / f"{code}.csv"

    if incremental and csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            last_date = existing["date"].max()
            if last_date.date() >= pd.to_datetime(end, format="%Y%m%d").date():
                logger.debug("%s 已是最新，无需更新", code)
                return
            start = last_date.strftime("%Y%m%d")
        except Exception:
            logger.exception("读取 %s 失败，将重新下载", csv_path)

    for attempt in range(1, 4):
        try:
            new_df = get_kline(code, start, end, "qfq", datasource, freq_code, tdx_dir)
            if new_df.empty:
                logger.debug("%s 无新数据", code)
                break
            new_df = validate(new_df)
            if csv_path.exists() and incremental:
                old_df = pd.read_csv(
                    csv_path,
                    parse_dates=["date"],
                    index_col=False
                )
                old_df = drop_dup_columns(old_df)
                new_df = drop_dup_columns(new_df)
                new_df = (
                    pd.concat([old_df, new_df], ignore_index=True)
                    .drop_duplicates(subset="date")
                    .sort_values("date")
                )
            new_df.to_csv(csv_path, index=False)
            break
        except Exception:
            logger.exception("%s 第 %d 次抓取失败", code, attempt)
            time.sleep(random.uniform(1, 3) * attempt)
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)


# ---------- 主入口 ---------- #

def main():
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--datasource", choices=["tushare", "akshare", "mootdx", "efinance"], default="mootdx", help="历史 K 线数据源")
    parser.add_argument("--tdx-dir", type=str, default=None, help="通达信安装目录，用于从本地文件读取数据，速度最快")
    parser.add_argument("--frequency", type=int, choices=list(_FREQ_MAP.keys()), default=4, help="K线频率编码，参见说明")
    parser.add_argument("--exclude-gem", type=lambda x: (str(x).lower() == 'true'), default=True, help="True则排除创业板/科创板/北交所")
    parser.add_argument("--min-mktcap", type=float, default=5e9, help="最小总市值（含），单位：元")
    parser.add_argument("--max-mktcap", type=float, default=float("+inf"), help="最大总市值（含），单位：元，默认无限制")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="并发线程数")
    args = parser.parse_args()

    # ---------- Token 处理 ---------- #
    if args.datasource == "tushare":
        ts_token = "8a29734db876d137f35ef50b7c742d473da1563fe4c6be0683c4a3c9"  # 在这里补充token
        ts.set_token(ts_token)

    # ---------- 日期解析 ---------- #
    start = dt.date.today().strftime("%Y%m%d") if args.start.lower() == "today" else args.start
    end = dt.date.today().strftime("%Y%m%d") if args.end.lower() == "today" else args.end

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 市值快照 & 股票池 ---------- #
    if args.tdx_dir:
        logger.info("检测到通达信目录，将从本地文件获取股票列表")
        mktcap_df = pd.DataFrame({"code": Reader.factory(market="std", tdxdir=args.tdx_dir).symbols, "mktcap": 0.0})
    elif args.datasource == "efinance":
        mktcap_df = _get_mktcap_efinance(out_dir)
    else:
        mktcap_df = _get_mktcap_ak()

    codes_from_filter = get_constituents(
        args.min_mktcap,
        args.max_mktcap,
        args.exclude_gem,
        mktcap_df=mktcap_df,
    )
    # 加上本地已有的股票，确保旧数据也能更新
    local_codes = [p.stem for p in out_dir.glob("*.csv")]
    codes = sorted(set(codes_from_filter) | set(local_codes))

    # ---------- 黑名单过滤 ---------- #
    try:
        with open("blacklist.json", "r", encoding="utf-8") as f:
            blacklist = json.load(f).get("data", [])
            if blacklist:
                before_count = len(codes)
                codes = [c for c in codes if c not in blacklist]
                logger.info("从黑名单中排除了 %d 只股票，剩余 %d 只", before_count - len(codes), len(codes))
    except FileNotFoundError:
        pass

    if not codes:
        logger.error("筛选结果为空，请调整参数！")
        sys.exit(1)

    logger.info(
        "开始抓取 %d 支股票 | 数据源:%s | 频率:%s | 日期:%s → %s",
        len(codes),
        args.datasource,
        _FREQ_MAP[args.frequency],
        start,
        end,
    )
    if args.tdx_dir:
        logger.info("使用本地通达信数据目录: %s", args.tdx_dir)


    # ---------- 多线程抓取 ---------- #
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                fetch_one,
                code,
                start,
                end,
                out_dir,
                True,
                args.datasource,
                args.frequency,
                args.tdx_dir,
            )
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            pass

    logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())


if __name__ == "__main__":
    main()
