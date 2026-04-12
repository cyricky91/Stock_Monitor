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
# gemini_key 會在 get_ai_comment 函數中被調用，不需額外初始化

# --- 股票清單與參數 ---
stock_ids = "OGN, SQFT"
time_period = "1y"
volume_spike_threshold = 2.0
bins_count = 70

def get_ai_comment(ticker, price, rsi, poc, action, advice):
    """標準 HTTP 請求版本：不依賴 SDK，穩定解決 404 問題"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "AI 分析未啟用 (缺少 API Key)"

    # 使用 v1beta 接口
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    作為資深量化分析師，針對股票 {ticker} 提供 100 字內繁體中文點評：
    數據：現價 ${price:.2f}, RSI {rsi:.1f}, 籌碼重心 ${poc:.2f}。
    主力行為：{action}。技術建議：{advice}。
    請務必包含：
    1. 短期建議買入區間（參考 POC）。
    2. 嚴格止損價位。
    直接給結論。
    """
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 300
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        return f"Gemini 診斷失敗: {str(e)[:50]}"

def run_diagnostic():
    tickers = [s.strip().upper() for s in re.split(r'[，,；;\s]+', stock_ids) if s.strip()]
    print(f"⏰ 啟動定時診斷：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_data = yf.download(tickers, period=time_period, interval="1d", progress=False, auto_adjust=True)
    
    for tk in tickers:
        try:
            if len(tickers) > 1:
                df = pd.DataFrame({'Close': all_data['Close'][tk], 'Volume': all_data['Volume'][tk]}).dropna()
            else:
                df = pd.DataFrame({'Close': all_data['Close'], 'Volume': all_data['Volume']}).dropna()

            if len(df) < 35: continue

            prices, vols = df['Close'].values, df['Volume'].values
            hist, bin_edges = np.histogram(prices, bins=bins_count, weights=vols)
            poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

            df['Price_Chg'] = df['Close'].diff()
            df['MF'] = np.where(df['Price_Chg'] > 0, df['Volume'], np.where(df['Price_Chg'] < 0, -df['Volume'], 0))
            df['Cum_MF'] = df['MF'].cumsum()

            delta = df['Price_Chg']
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD'] = df['Close'].rolling(window=20).std()
            df['Upper'] = df['MA20'] + (df['STD'] * 2)
            df['Lower'] = df['MA20'] - (df['STD'] * 2)

            latest_p = float(df['Close'].iloc[-1])
            pct_chg = ((latest_p / df['Close'].iloc[-2]) - 1) * 100
            vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].tail(20).mean())
            cur_rsi = float(df['RSI'].iloc[-1])
            
            main_action = "🔘 籌碼縮量整理"
            if vol_ratio >= volume_spike_threshold:
                if pct_chg > 1.8: main_action = "🔥 主力放量進場"
                elif pct_chg < -1.8: main_action = "😱 主力放量派發"
            
            advice = "等待訊號"
            if latest_p > poc_price:
                advice = "✅ 多頭強勢，回調支撐可佈局" if cur_rsi < 70 else "⚠️ 超買警告，建議分批獲利"
            else:
                advice = "🔵 超跌區域，觀察反彈" if cur_rsi < 30 else "📉 弱勢壓制，建議觀望"

            ai_note = get_ai_comment(tk, latest_p, cur_rsi, poc_price, main_action, advice)

            report = (
                f"📊 **{tk} 定時診斷報告**\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"💰 **現價**：${latest_p:.2f} ({pct_chg:+.2f}%)\n"
                f"📍 **籌碼重心**：${poc_price:.2f}\n"
                f"🔍 **主力行為**：{main_action}\n"
                f"💡 **交易建議**：**{advice}**\n\n"
                f"🤖 **Gemini AI 點評**：\n{ai_note}\n"
                f"━━━━━━━━━━━━━━━━━━"
            )

            if telegram_token and chat_id:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [2, 1]})
                plot_df = df.tail(120)
                
                ax1.plot(plot_df.index, plot_df['Close'], color='black', label='Price')
                ax1.axhline(poc_price, color='red', ls='--', alpha=0.6, label=f'POC: {poc_price:.2f}')
                ax1.fill_between(plot_df.index, plot_df['Upper'], plot_df['Lower'], color='gray', alpha=0.15, label='BB Bands')
                ax1.set_title(f"{tk} Trend Analysis"); ax1.legend(loc='upper left')

                ax2.fill_between(plot_df.index, plot_df['Cum_MF'], color='purple', alpha=0.1)
                ax2.plot(plot_df.index, plot_df['Cum_MF'], color='purple', label='Money Flow Strength')
                ax2.legend(loc='upper left')

                plt.tight_layout()
                
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage", 
                             data={"chat_id": chat_id, "text": report, "parse_mode": "Markdown"})
                
                buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
                requests.post(f"https://api.telegram.org/bot{telegram_token}/sendPhoto", 
                             data={"chat_id": chat_id}, files={"photo": buf})
                
                plt.close(fig)
                print(f"✅ {tk} 深度報告已送出")

        except Exception as e:
            print(f"❌ {tk} 失敗: {e}")

if __name__ == "__main__":
    run_diagnostic()
