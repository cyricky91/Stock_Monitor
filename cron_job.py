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

# 基礎環境設定
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取環境變數
telegram_token = os.getenv("TELEGRAM_TOKEN")
chat_id = os.getenv("CHAT_ID")
send_to_tg = True

# 2. 監控參數
stock_ids = "OGN, SQFT, TLX, OKLO, MRLN, MP, VST, ACLS, IONQ, FIGR, PLTR, RANI" 
time_period = "1y"
volume_spike_threshold = 2.0
bins_count = 70

def run_diagnostic():
    tickers = [s.strip().upper() for s in re.split(r'[，,；;\s]+', stock_ids) if s.strip()]
    print(f"⏰ 啟動終極診斷：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_data = yf.download(tickers, period=time_period, interval="1d", progress=False, auto_adjust=True)
    
    for tk in tickers:
        try:
            if len(tickers) > 1:
                df = pd.DataFrame({'Close': all_data['Close'][tk], 'Volume': all_data['Volume'][tk]}).dropna()
            else:
                df = pd.DataFrame({'Close': all_data['Close'], 'Volume': all_data['Volume']}).dropna()

            if len(df) < 35: continue

            # --- A. 技術指標與資金流計算 ---
            # 1. 籌碼重心 (POC)
            prices, vols = df['Close'].values, df['Volume'].values
            hist, bin_edges = np.histogram(prices, bins=bins_count, weights=vols)
            poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

            # 2. 累計資金流 (Cumulative Money Flow)
            df['Price_Chg'] = df['Close'].diff()
            df['MF'] = np.where(df['Price_Chg'] > 0, df['Volume'], np.where(df['Price_Chg'] < 0, -df['Volume'], 0))
            df['Cum_MF'] = df['MF'].cumsum()

            # 3. RSI & 布林通道
            delta = df['Price_Chg']
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['MA20'] + (df['STD'] * 2)
            df['Lower'] = df['MA20'] - (df['STD'] * 2)

            # --- B. 主力行為與策略判定 ---
            latest_p = df['Close'].iloc[-1]
            pct_chg = ((latest_p / df['Close'].iloc[-2]) - 1) * 100
            vol_ratio = df['Volume'].iloc[-1] / df['Volume'].tail(20).mean()
            cur_rsi = df['RSI'].iloc[-1]
            
            # 判定主力行為
            main_action = "🔘 籌碼縮量整理"
            if vol_ratio >= volume_spike_threshold:
                if pct_chg > 1.8: main_action = "🔥 主力放量進場 (強烈看漲)"
                elif pct_chg < -1.8: main_action = "😱 主力放量派發 (高度戒備)"
            
            # 策略建議
            advice = "等待訊號"
            if latest_p > poc_price:
                advice = "✅ 多頭強勢，回調 POC 支撐可佈局" if cur_rsi < 70 else "⚠️ 超買警告，建議獲利了結"
            else:
                advice = "🔵 超跌機會，觀察反彈" if cur_rsi < 30 else "📉 弱勢壓制，建議觀望"

            # --- C. 報告內容 ---
            report = (
                f"📊 **{tk} 深度分析報告**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"💰 **現價**：${latest_p:.2f} ({pct_chg:+.2f}%)\n"
                f"📍 **籌碼重心**：${poc_price:.2f}\n"
                f"🔍 **主力行為**：{main_action}\n"
                f"💡 **交易建議**：**{advice}**\n\n"
                f"🎯 **操作參考**：\n"
                f"   - 買入支撐位：${poc_price:.2f}\n"
                f"   - 止損參考位：${poc_price * 0.95:.2f}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )

            # --- D. 雙子圖繪製與發送 ---
            if send_to_tg and telegram_token and chat_id:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [2, 1]})
                
                # 上圖：價格與布林帶
                ax1.plot(df.indextk120:], df['Close'].tail(120), color='black', label='Price')
                ax1.axhline(poc_price, color='red', ls='--', label=f'POC: {poc_price:.2f}')
                ax1.fill_between(df.index[-120:], df['Upper'].tail(120), df['Lower'].tail(120), color='gray', alpha=0.2, label='BB Bands')
                ax1.set_title(f"{tk} Trend & Cost Structure"); ax1.legend(loc='upper left')

                # 下圖：累計資金流
                ax2.fill_between(df.index[-120:], df['Cum_MF'].tail(120), color='purple', alpha=0.1)
                ax2.plot(df.index[-120:], df['Cum_MF'].tail(120), color='purple', label='Cumulative Money Flow')
                ax2.set_title("Money Flow Intelligence"); ax2.legend(loc='upper left')

                plt.tight_layout()
                
                # 發送文字與圖片
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage", data={"chat_id": chat_id, "text": report, "parse_mode": "Markdown"})
                buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendPhoto", data={"chat_id": chat_id}, files={"photo": buf})
                plt.close(fig)
                print(f"✅ {tk} 深度報告已送出")

        except Exception as e:
            print(f"❌ {tk} 失敗: {e}")

if __name__ == "__main__":
    run_diagnostic()
