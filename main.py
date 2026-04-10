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

# 1. 讀取環境變數 (請確保 GitHub Secrets 已設定)
telegram_token = os.getenv("TELEGRAM_TOKEN")
chat_id = os.getenv("CHAT_ID")
send_to_tg = True

# 2. 監控參數
stock_ids = "AAPL, NVDA, TSLA, MSFT, FORM, SKYQ, OGN" # 可自行增減
time_period = "1y"
volume_threshold = 2.0
bins_count = 70

def run_diagnostic():
    tickers = [s.strip().upper() for s in re.split(r'[，,；;\s]+', stock_ids) if s.strip()]
    print(f"⏰ 啟動策略診斷：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 下載數據
    all_data = yf.download(tickers, period=time_period, interval="1d", progress=False, auto_adjust=True)
    
    for tk in tickers:
        try:
            # 數據對齊
            if len(tickers) > 1:
                df = pd.DataFrame({'Close': all_data['Close'][tk], 'Volume': all_data['Volume'][tk]}).dropna()
            else:
                df = pd.DataFrame({'Close': all_data['Close'], 'Volume': all_data['Volume']}).dropna()

            if len(df) < 30: continue

            # --- A. 技術指標計算 ---
            # 1. 籌碼重心 (POC)
            prices, vols = df['Close'].values, df['Volume'].values
            hist, bin_edges = np.histogram(prices, bins=bins_count, weights=vols)
            poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

            # 2. RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            cur_rsi = df['RSI'].iloc[-1]

            # 3. 布林通道 (20, 2)
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['MA20'] + (df['STD'] * 2)
            df['Lower'] = df['MA20'] - (df['STD'] * 2)
            up_band, lo_band = df['Upper'].iloc[-1], df['Lower'].iloc[-1]

            # --- B. 智能策略判定 ---
            latest_p = df['Close'].iloc[-1]
            pct_chg = ((latest_p / df['Close'].iloc[-2]) - 1) * 100
            vol_ratio = df['Volume'].iloc[-1] / df['Volume'].tail(20).mean()
            
            # 核心邏輯
            advice = "🔘 觀望：等待訊號"
            signal_color = "⚪"
            
            if latest_p > poc_price:
                if cur_rsi > 70:
                    advice = "⚠️ 超買提示：接近頂部，建議逢高分批減磅"; signal_color = "🟡"
                elif latest_p > up_band:
                    advice = "🚀 強勢突破：突破布林上軌，持股待漲但勿追高"; signal_color = "🟢"
                else:
                    advice = "✅ 多頭佔優：現價高於成本重心，回調至 POC 可加倉"; signal_color = "🟢"
            else:
                if cur_rsi < 30:
                    advice = "🔥 底部機會：嚴重超賣 + 跌破下軌，隨時反彈"; signal_color = "🔵"
                elif latest_p < lo_band:
                    advice = "😱 恐慌殺跌：股價跌穿布林下軌，暫避風頭"; signal_color = "🔴"
                else:
                    advice = "📉 弱勢整理：受壓於成本重心，暫不宜進場"; signal_color = "🔴"

            # 獲利與止損建議
            take_profit = poc_price * 1.15
            stop_loss = poc_price * 0.95

            # --- C. 報告格式化 ---
            report = (
                f"📊 **{tk} 策略診斷報告**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"💰 **現價**：${latest_p:.2f} ({pct_chg:+.2f}%)\n"
                f"📍 **籌碼重心 (POC)**：${poc_price:.2f}\n"
                f"⚖️ **RSI 指標**：{cur_rsi:.1f} ({'超買' if cur_rsi>70 else '超賣' if cur_rsi<30 else '中性'})\n\n"
                f"💡 **交易建議**：{signal_color} **{advice}**\n\n"
                f"🎯 **操作參考**：\n"
                f"   - 建議買入位：${poc_price:.2f} (支撐)\n"
                f"   - 止損參考位：${stop_loss:.2f}\n"
                f"   - 獲利目標位：${take_profit:.2f}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )

            # --- D. 發送至 Telegram ---
            if send_to_tg and telegram_token and chat_id:
                # 繪圖
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df.index[-100:], df['Close'].tail(100), label='Price', color='black')
                ax.axhline(poc_price, color='red', ls='--', alpha=0.7, label='POC')
                ax.fill_between(df.index[-100:], df['Upper'].tail(100), df['Lower'].tail(100), color='gray', alpha=0.2, label='BB Bands')
                ax.set_title(f"{tk} Strategic Analysis")
                ax.legend()

                # 發送文字
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage", 
                              data={"chat_id": chat_id, "text": report, "parse_mode": "Markdown"})
                
                # 發送圖片
                buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendPhoto", 
                              data={"chat_id": chat_id}, files={"photo": buf})
                plt.close(fig)
                print(f"✅ {tk} 診斷報告已送出")

        except Exception as e:
            print(f"❌ {tk} 失敗: {e}")

if __name__ == "__main__":
    run_diagnostic()
