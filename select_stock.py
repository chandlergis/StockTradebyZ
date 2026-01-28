from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具 ----------

def load_single_csv(fp: Path) -> Tuple[str, pd.DataFrame] | None:
    """读取单个CSV文件的辅助函数，用于并发加载"""
    try:
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        return fp.stem, df
    except Exception as e:
        # 在多进程中直接打印日志可能导致乱序，这里简单略过或用print
        return None


def load_data_concurrent(data_dir: Path, codes: List[str]) -> Dict[str, pd.DataFrame]:
    """并发加载股票数据"""
    frames: Dict[str, pd.DataFrame] = {}
    files = []
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if fp.exists():
            files.append(fp)
        else:
            logger.warning("%s 不存在，跳过", fp.name)

    if not files:
        return frames

    # 使用进程池并发读取
    # max_workers 默认使用 CPU 核数
    with ProcessPoolExecutor() as executor:
        future_to_fp = {executor.submit(load_single_csv, fp): fp for fp in files}
        
        for future in as_completed(future_to_fp):
            fp = future_to_fp[future]
            try:
                result = future.result()
                if result:
                    code, df = result
                    frames[code] = df
            except Exception as e:
                logger.warning("读取文件 %s 失败: %s", fp, e)
    
    return frames


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


def worker_check_stock(
    code: str, 
    df: pd.DataFrame, 
    selectors: List[Tuple[str, Any]], 
    date: pd.Timestamp
) -> Tuple[str, List[str]]:
    """
    工作进程函数：对单只股票运行所有 Selectors
    返回: (code, [matched_alias_1, matched_alias_2, ...])
    """
    matched_aliases = []
    
    # 预先切片数据，减少传给 selector 的数据量（虽然这里是在内存中切片）
    # 大多数 selector 需要的数据都在最近 200 天以内
    # 但为了保险，传递所有数据或根据最大需求切片
    # 为了兼容性，我们传递原始 df 的切片（截止到 trade_date）
    hist = df[df["date"] <= date]
    if hist.empty:
        return code, []

    # 构造单只股票的字典，适配 selector.select 接口
    data_shard = {code: hist}

    for alias, selector in selectors:
        try:
            # 调用 selector.select
            # 注意：这里会重复进行一些 df 操作，但节省了跨进程传递多份数据的开销
            picks = selector.select(date, data_shard)
            if picks:
                matched_aliases.append(alias)
        except Exception:
            # 策略执行出错，忽略
            pass
            
    return code, matched_aliases


# ---------- 主函数 ----------

def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json (Concurrent)")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    p.add_argument("--workers", type=int, default=None, help="并发进程数，默认自动设置")
    args = p.parse_args()

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))
    
    # 实例化所有启用的 Selectors
    active_selectors: List[Tuple[str, Any]] = []
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue
        try:
            alias, selector = instantiate_selector(cfg)
            active_selectors.append((alias, selector))
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)

    if not active_selectors:
        logger.error("没有可用的 Selector，退出")
        sys.exit(0)

    # --- 加载行情 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("数据目录 %s 不存在", data_dir)
        sys.exit(1)

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if args.tickers.lower() == "all"
        else [c.strip() for c in args.tickers.split(",") if c.strip()]
    )
    if not codes:
        logger.error("股票池为空！")
        sys.exit(1)

    logger.info("正在并发加载行情数据...")
    data = load_data_concurrent(data_dir, codes)
    if not data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)
    logger.info("已加载 %d 只股票行情", len(data))

    trade_date = (
        pd.to_datetime(args.date)
        if args.date
        else max(df["date"].max() for df in data.values())
    )
    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 并发执行选股 ---
    logger.info("正在并发执行 %d 个选股策略...", len(active_selectors))
    
    # 结果容器：{alias: [code1, code2, ...]}
    results: Dict[str, List[str]] = {alias: [] for alias, _ in active_selectors}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 提交任务
        future_to_code = {
            executor.submit(worker_check_stock, code, df, active_selectors, trade_date): code
            for code, df in data.items()
        }
        
        # 收集结果
        # 使用 as_completed 可以实时处理完成的任务
        count = 0
        total = len(data)
        for future in as_completed(future_to_code):
            try:
                code, matched_aliases = future.result()
                for alias in matched_aliases:
                    if alias in results:
                        results[alias].append(code)
                
                count += 1
                if count % 500 == 0:
                    logger.info("进度: %d/%d", count, total)
                    
            except Exception as e:
                logger.error("处理股票失败: %s", e)

    # --- 输出结果 ---
    for alias, _ in active_selectors:
        picks = sorted(results.get(alias, []))
        
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        logger.info("%s", ", ".join(picks) if picks else "无符合条件股票")


if __name__ == "__main__":
    # 确保在 Windows/macOS 上正确行为，Linux 上通常默认 fork
    # 如果遇到 pickling 问题可能需要 'spawn'，但 fork 性能更好
    # 这里保持默认，除非显式需要
    try:
        mp.set_start_method('fork')
    except Exception:
        pass
    main()
