
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import cosine
from tqdm import tqdm

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("select_results_dtw.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("dtw_selector")


# --- 数据处理与计算 ---

def min_max_normalize(series: np.ndarray) -> np.ndarray:
    """对一维数组进行Min-Max归一化"""
    if series.min() == series.max():
        return np.zeros_like(series)
    return (series - series.min()) / (series.max() - series.min())


def get_template_series(
    stock_code: str, start_date: str, end_date: str, data_dir: Path
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """加载并处理模板股票的K线数据"""
    fp = data_dir / f"{stock_code}.csv"
    if not fp.exists():
        logger.error(f"模板股票 {stock_code} 的数据文件不存在于 {data_dir}")
        return None

    df = pd.read_csv(fp, parse_dates=["date"])
    template_df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if template_df.empty:
        logger.error(f"在指定日期范围 {start_date} - {end_date} 内找不到模板股票 {stock_code} 的数据")
        return None

    p_template = min_max_normalize(template_df["close"].values)
    v_template = min_max_normalize(template_df["volume"].values)
    window_size = len(template_df)

    logger.info(f"成功加载模板 {stock_code} ({start_date} to {end_date})，窗口长度: {window_size}")
    return p_template, v_template, window_size


def process_stock(
    stock_file: Path,
    p_template: np.ndarray,
    v_template: np.ndarray,
    window_size: int,
    price_weight: float,
    trade_date: pd.Timestamp,
    dtw_radius: int,
) -> dict | None:
    """
    对单个股票在指定交易日的形态进行DTW相似度匹配。
    只比较截止到 trade_date 的最后一个窗口。
    """
    try:
        df = pd.read_csv(stock_file, parse_dates=["date"])
        
        # 过滤出指定日期之前的数据
        df_recent = df[df["date"] <= trade_date].copy()

        # 如果数据不足一个窗口，则跳过
        if len(df_recent) < window_size:
            return None

        # 提取最新的一个窗口作为候选
        candidate_window = df_recent.tail(window_size)
        p_candidate_raw = candidate_window["close"].values
        v_candidate_raw = candidate_window["volume"].values

        # 数据质量检查
        if np.any(p_candidate_raw <= 0) or np.any(v_candidate_raw < 0):
            return None

        # 归一化
        p_candidate_norm = min_max_normalize(p_candidate_raw)
        v_candidate_norm = min_max_normalize(v_candidate_raw)

        # 直接计算DTW距离
        dist_p_dtw, _ = fastdtw(p_template, p_candidate_norm, dist=2, radius=dtw_radius)
        dist_v_dtw, _ = fastdtw(v_template, v_candidate_norm, dist=2, radius=dtw_radius)

        combined_dtw_dist = price_weight * dist_p_dtw + (1 - price_weight) * dist_v_dtw
        
        # 匹配的日期是窗口的最后一天
        match_date = candidate_window["date"].iloc[-1]

        return {
            "dtw_dist": combined_dtw_dist,
            "date": match_date.strftime('%Y-%m-%d'),
            "code": stock_file.stem,
        }

    except Exception as e:
        # logger.warning(f"处理 {stock_file.name} 时出错: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="基于DTW和余弦相似度进行股票形态匹配")
    parser.add_argument("--template-stock", required=True, help="模板股票代码, e.g., 000001")
    parser.add_argument("--start-date", required=True, help="模板起始日期, 格式 YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="模板结束日期, 格式 YYYYMMDD")
    parser.add_argument("--data-dir", default="./data", help="股票CSV数据目录")
    parser.add_argument("--date", help="选股日期, 格式 YYYYMMDD, 默认为最新数据日期")
    parser.add_argument("--workers", type=int, default=10, help="并发进程数")
    parser.add_argument("--price-weight", type=float, default=0.7, help="价格形态的权重 (0.0-1.0)")
    parser.add_argument("--similarity-threshold", type=float, default=80.0, help="相似度阈值 (e.g., 80 for 80%%)")
    parser.add_argument("--dtw-radius", type=int, default=5, help="DTW匹配半径, 数值越小越严格")
    parser.add_argument("--decay-k", type=float, default=0.5, help="相似度衰减系数, 数值越大打分越严格")
    parser.add_argument("--top-n", type=int, default=20, help="显示Top N个结果用于调参")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"数据目录 {data_dir} 不存在")
        sys.exit(1)

    # 1. 获取模板序列
    template_info = get_template_series(args.template_stock, args.start_date, args.end_date, data_dir)
    if not template_info:
        sys.exit(1)
    p_template, v_template, window_size = template_info

    # 2. 确定选股日期
    if args.date:
        trade_date = pd.to_datetime(args.date)
    else:
        # 默认使用数据全集中的最新日期
        all_dates = [pd.read_csv(f, usecols=['date'])['date'].max() for f in data_dir.glob("*.csv")]
        trade_date = pd.to_datetime(max(all_dates))
        logger.info(f"未指定 --date, 将使用所有数据中的最新日期: {trade_date.date()}")

    # 3. 扫描股票池
    all_stock_files = [f for f in data_dir.glob("*.csv") if f.stem != args.template_stock]
    results = []

    logger.info(f"开始扫描 {len(all_stock_files)} 支股票，选股日期: {trade_date.date()}，DTW半径: {args.dtw_radius}，使用 {args.workers} 个并发进程...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_stock, f, p_template, v_template, window_size, args.price_weight, trade_date, args.dtw_radius): f
            for f in all_stock_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="形态匹配中"):
            res = future.result()
            if res and res.get("dtw_dist") is not None:
                results.append(res)

    if not results:
        logger.info("没有找到任何形态相似的股票。")
        sys.exit(0)

    # 3. 结果排序与输出
    # 使用指数衰减将距离转换为相似度，对较大距离进行更强的惩罚
    # k值用于控制衰减速度，可根据实际情况调整
    k = args.decay_k 
    for r in results:
        r["similarity"] = 100 * np.exp(-k * r["dtw_dist"])

    # 按相似度从高到低排序
    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    
    # 过滤并输出
    final_picks = [r for r in sorted_results if r["similarity"] >= args.similarity_threshold]
    top_n_to_display = sorted_results[:args.top_n]

    logger.info("")
    logger.info("="*20 + " DTW形态匹配结果 " + "="*20)
    logger.info(f"模板: {args.template_stock} ({args.start_date} - {args.end_date})")
    logger.info(f"DTW半径: {args.dtw_radius}, 衰减系数k: {args.decay_k}")
    logger.info(f"价格权重: {args.price_weight}, 成交量权重: {1-args.price_weight:.1f}")
    logger.info(f"相似度阈值: >={args.similarity_threshold}%%")
    logger.info(f"符合条件(高于阈值)的股票数: {len(final_picks)}")
    logger.info("-" * 68)
    logger.info(f"--- 显示相似度最高的 Top {len(top_n_to_display)} 支股票 (带'*'为达到阈值) ---")

    if not top_n_to_display:
        logger.info("在当前参数下，无任何匹配结果。")
    else:
        logger.info(f"{'排名':<5}{'股票代码':<12}{'匹配日期':<15}{'相似度':<12}{'DTW距离':<10}")
        for i, r in enumerate(top_n_to_display):
            marker = " *" if r['similarity'] >= args.similarity_threshold else ""
            logger.info(f"{i+1:<5}{r['code']:<12}{r['date']:<15}{r['similarity']:.2f}%%{marker:<2}{r['dtw_dist']:.4f}")
    logger.info("=" * 68)


if __name__ == "__main__":
    main()
