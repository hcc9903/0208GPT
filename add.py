import akshare as ak
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import concurrent.futures
import logging
import sys
import argparse
from typing import List, Dict, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def get_limit_price(code: str, prev_close: float) -> float:
    """
    计算涨停价（考虑误差修正）
    """
    ratio = 1.20 if code.startswith(("30", "68")) else 1.10
    return round(prev_close * ratio + 0.0001, 2)

def process_stock(stock: Dict[str, str], target_date: str, start_date: str) -> Optional[Dict]:
    """
    分析单只股票是否符合涨停逻辑
    """
    code = stock['code']
    name = stock['name']
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, 
            period="daily", 
            start_date=start_date, 
            adjust="qfq"
        )
        
        if df.empty or len(df) < 2:
            return None
        
        df['日期'] = df['日期'].astype(str).str.replace('-', '')
        
        if target_date not in df['日期'].values:
            return None
            
        target_indices = df[df['日期'] == target_date].index
        idx_t5 = target_indices[0]
        if idx_t5 == 0: return None 
        
        row_t5 = df.loc[idx_t5]
        row_prev = df.loc[idx_t5 - 1]
        row_latest = df.iloc[-1]
        
        limit_price = get_limit_price(code, row_prev['收盘'])
        
        if row_t5['最高'] >= limit_price:
            t5_pct = (row_t5['收盘'] - row_prev['收盘']) / row_prev['收盘'] * 100
            period_pct = (row_latest['收盘'] - row_t5['收盘']) / row_t5['收盘'] * 100
            period_turnover = df.loc[idx_t5:, '换手率'].sum()
            
            return {
                "代码": code,
                "名称": name,
                "区间涨幅%": round(period_pct, 2),
                "区间换手%": round(period_turnover, 2),
                "T-5涨幅%": round(t5_pct, 2),
                "T-5状态": "涨停" if row_t5['收盘'] >= limit_price else "曾涨停",
                "当前价": row_latest['收盘']
            }
    except Exception:
        return None
    return None

def main():
    parser = argparse.ArgumentParser(description="A股涨停回测选股工具 (Optimized)")
    parser.add_argument('--date', type=str, default=os.getenv('TARGET_DATE', "20260203"), help='目标分析日期 YYYYMMDD')
    parser.add_argument('--start', type=str, default=os.getenv('START_DATE', "20260120"), help='起始分析日期 YYYYMMDD')
    parser.add_argument('--workers', type=int, default=int(os.getenv('MAX_WORKERS', 15)), help='并行线程数')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info(f"🚀 启动重构版选股工具 | 目标: {args.date} | 线程: {args.workers}")
    logger.info("="*60)
    
    try:
        # 使用更全的实时行情接口获取全市场代码
        df_all = ak.stock_zh_a_spot_em()
        stock_list = df_all[df_all['代码'].str.startswith(('00', '60', '300', '688'))]
        stock_list = stock_list.rename(columns={'代码': 'code', '名称': 'name'})[['code', 'name']].to_dict('records')
        logger.info(f"📦 成功加载 {len(stock_list)} 只股票数据")
    except Exception as e:
        logger.error(f"❌ 加载股票列表异常: {e}")
        return

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_stock, s, args.date, args.start): s for s in stock_list}
        with tqdm(total=len(stock_list), desc="分析中", unit="只") as pbar:
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)
                pbar.update(1)

    if results:
        final_df = pd.DataFrame(results).sort_values(by="区间涨幅%", ascending=False)
        logger.info(f"✅ 完成！符合条件数量: {len(results)}")
        
        print("\n" + final_df.to_string(index=False))
        
        output_file = f"results_{args.date}.csv"
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"💾 结果保存至: {output_file}")
        
        if os.getenv('GITHUB_STEP_SUMMARY'):
            with open(os.getenv('GITHUB_STEP_SUMMARY'), 'a', encoding='utf-8') as f:
                f.write(f"### � 选股报告 ({args.date})\n")
                f.write(f"- 扫描标的总数: {len(stock_list)}\n")
                f.write(f"- 触发涨停/曾涨停: {len(results)}\n\n")
                f.write(final_df.head(20).to_markdown(index=False) + "\n")
    else:
        logger.info("⚠️ 当前条件下未发现符合标的。")

if __name__ == "__main__":
    main()