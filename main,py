import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import warnings
import re
from datetime import datetime

# 基礎設定
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 參數設定 (從環境變數讀取敏感資訊)
# 請確保在 GitHub Secrets 裡設定了 TELEGRAM_TOKEN 和 CHAT_ID
telegram_token = os.getenv("TELEGRAM_TOKEN")
chat_id = os.getenv("CHAT_ID")
send_to_tg = True

# 股票清單 (可以直接在這邊修改)
stock_ids = "FORM, SKYQ, OGN, MRLN, TLX, AXTI, SPIR, AGX, REPL, SQFT"
time_period = "1y"
volume_spike_threshold = 2.0
bins_count = 60

def run_diagnostic():
    raw_ids = re.split(r'[，,；;\s]+', stock_ids)
    clean_ids = [s.strip().upper() for s in raw_ids if s.strip()]
    
    print(f"⏰ 開始診斷任務：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 批量下載
    all_data = yf.download(clean_ids, period=time_period, interval="1d", progress=False, auto_adjust=True)
    
    for tk in clean_ids:
        try:
            # 數據對齊與降維 (修復 ValueError)
            if len(clean_ids) > 1:
                df = pd.DataFrame({'Close': all_data['Close'][tk], 'Volume': all_data['Volume'][tk]}).dropna()
            else:
                df = pd.DataFrame({'Close': all_data['Close'], 'Volume': all_data['Volume']}).dropna()

            if df.empty or len(df) < 20:
                print(f"⚠️ {tk} 數據不足，跳過。")
                continue

            # 籌碼分布計算
            prices, vols = df['Close'].values, df['Volume'].values
            hist, bin_edges = np.histogram(prices, bins=bins_count, weights=vols)
            poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

            # 診斷指標
            latest_p = float(df['Close'].iloc[-1])
            pct_chg = ((latest_p / df['Close'].iloc[-2]) - 1) * 100
            vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].tail(20).mean())

            # 邏輯判定
            status = "🔴 弱勢反壓" if latest_p < poc_price else "🟢 強勢支撐"
            action = "🔘 橫盤整理"
            if vol_ratio >= volume_spike_threshold:
                if pct_chg > 1.8: action = "🔥 主力建倉"
                elif pct_chg < -1.8: action = "😱 主力派發"

            report = (f"🔍 **{tk} 診斷結果**\n"
                      f"💰 現價：${latest_p:.2f} ({pct_chg:+.2f}%)\n"
                      f"📍 籌碼重心：${poc_price:.2f}\n"
                      f"💡 主力行為：{action}")

            print(f"✅ {tk} 診斷完成：{action}")

            # 發送 Telegram (文字 + 圖表)
            if send_to_tg and telegram_token and chat_id:
                # 繪圖 (無介面環境需儲存至 Buffer)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df['Close'], label='Price')
                ax.axhline(poc_price, color='red', ls='--', label='POC')
                ax.set_title(f"{tk} Trend")
                
                # 發送文字
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage", 
                              data={"chat_id": chat_id, "text": report, "parse_mode": "Markdown"})
                
                # 發送圖片
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendPhoto", 
                              data={"chat_id": chat_id}, files={"photo": buf})
                plt.close(fig) # 釋放內存

        except Exception as e:
            print(f"❌ {tk} 發生錯誤: {e}")

if __name__ == "__main__":
    run_diagnostic()
