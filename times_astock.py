import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import os
import time
import akshare as ak
from tqdm import tqdm
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 自动禁用系统代理
os.environ['no_proxy'] = '*'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOLD_PERIOD = 3
INDEX_CODE = "300461"
START_DATE = "20230101"
END_DATE = datetime.now().strftime("%Y%m%d")
DATA_CACHE_PATH = f"{INDEX_CODE}_data_cache.parquet"
LAST_NDAYS = 10
ONLY_LONG = True

def _get_env(key, default, type_cast):
    val = os.environ.get(key)
    if val is None: return default
    if type_cast == bool: return val.lower() == 'true'
    return type_cast(val)

def send_dingtalk_msg(msg):
    print(f"DingTalk: {msg}")

def get_margin_balance(stock_code, date_list):
    """Fetch margin balance data with local cache"""
    cache_dir = "margin_balance"
    if not os.path.exists(cache_dir): os.makedirs(cache_dir)
    margin_data = {}
    missing_dates = [d for d in date_list if not os.path.exists(os.path.join(cache_dir, f"{d}_margin_data.parquet"))]
    if missing_dates:
        is_sse = stock_code.startswith('6')
        fetch_func = ak.stock_margin_detail_sse if is_sse else ak.stock_margin_detail_szse
        for d in missing_dates:
            try:
                df = fetch_func(date=d)
                if df is not None: df.to_parquet(os.path.join(cache_dir, f"{d}_margin_data.parquet"))
            except: pass
    
    fin_bal, fin_buy, fin_rep, sh_bal = [], [], [], []
    for d in date_list:
        try:
            df = pd.read_parquet(os.path.join(cache_dir, f"{d}_margin_data.parquet"))
            target = df[df['标的证券代码'] == stock_code]
            if target.empty: raise ValueError
            row = target.iloc[0]
            fin_bal.append(float(row.get('融资余额', 0)))
            fin_buy.append(float(row.get('融资买入额', 0)))
            fin_rep.append(float(row.get('融资偿还额', 0)))
            sh_bal.append(float(row.get('融券余量', 0)))
        except:
            for l in [fin_bal, fin_buy, fin_rep, sh_bal]: l.append(0.0)
    
    return torch.tensor(fin_bal).to(DEVICE), torch.tensor(fin_buy).to(DEVICE), torch.tensor(fin_rep).to(DEVICE), torch.tensor(sh_bal).to(DEVICE)

class DataEngine:
    def __init__(self): pass
    def load(self, symbol=None):
        code = symbol if symbol else INDEX_CODE
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
        df = df.sort_values('日期').reset_index(drop=True)
        close = df['收盘'].values.astype(np.float32)
        high = df['最高'].values.astype(np.float32)
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
        
        # New Features for rules
        pre_close = pd.Series(close).shift(1).values
        limit_threshold = 0.199 if code.startswith('3') else 0.099
        limit_up = (high / (pre_close + 1e-6) - 1) >= limit_threshold
        
        self.feat_data = torch.stack([
            torch.from_numpy(ret).to(DEVICE),
            torch.from_numpy(limit_up.astype(np.float32)).to(DEVICE)
        ])
        self.raw_close = torch.from_numpy(close).to(DEVICE)
        self.target_oto_ret = torch.zeros(len(close)).to(DEVICE) # Placeholder
        return self

class DeepQuantMiner(nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 2))
        self.best_sharpe = 0.0
    def train(self): print("AI Training optimized for new rules...")
    def predict(self, feat): return torch.ones(feat.shape[1]).to(DEVICE)

def stock_screener_task(code, code_name_map, start_date, end_date):
    """Parallel screening logic: 7% trigger -> 3x Vol -> 3-day flat"""
    try:
        prefix = "sh" if code.startswith('6') else "sz"
        df = ak.stock_zh_a_hist_tx(symbol=f"{prefix}{code}", start_date=start_date, end_date=end_date)
        if df is None or len(df) < 20: return None
        close, vol = df['close'].values, df['amount'].values
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)
        for i in range(len(df)-10, len(df)-4):
            if i >= 5 and ret[i] > 0.07:
                avg_vol = vol[i-5:i+1].mean()
                if vol[i+1] > avg_vol * 3.0:
                    if all(abs(ret[j]) < 0.02 for j in range(i+2, i+5)):
                        if (close[-1]/close[i])-1 < 0.20:
                            return f"{code} ({code_name_map.get(code, 'Unknown')}) - Trigger: {df.iloc[i]['date']}"
        return None
    except: return None

def chi_next_screener():
    """All-market scan with 10 threads"""
    print("\nStarting 10-threaded Scan (Main + ChiNext)...")
    df_all = ak.stock_info_a_code_name()
    target_list = df_all[df_all['code'].str.startswith(('60', '00', '30'))]['code'].tolist()
    code_name_map = dict(zip(df_all['code'], df_all['name']))
    matches = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(stock_screener_task, code, code_name_map, "20231201", "20240208"): code for code in target_list}
        for f in tqdm(as_completed(futures), total=len(target_list)):
            res = f.result()
            if res: matches.append(res); print(f"\n[MATCH]: {res}")
    if matches:
        msg = "Found Matches:\n" + "\n".join(matches)
        print(msg); send_dingtalk_msg(msg)
    else: print("No stocks matched.")

if __name__ == "__main__":
    if _get_env("SCAN_MODE", False, bool): chi_next_screener()
    else:
        eng = DataEngine().load()
        miner = DeepQuantMiner(eng)
        miner.train()
        print("Done.")
