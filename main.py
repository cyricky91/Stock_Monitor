import logging
import re
import io
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# 1. 基礎設定
TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

# 分析參數
BINS_COUNT = 80
VOL_THRESHOLD = 2.0

# 設定日誌
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def get_ai_comment(ticker, price, rsi, poc, action, advice, rr_ratio):
    """呼叫 DeepSeek API 生成點評"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用 (請檢查 DEEPSEEK_API_KEY 設定)"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"""
    作為專業資深交易員，請對股票 {ticker} 進行診斷：
    【關鍵數據】
    - 現價：{price}
    - 籌碼重心 (POC)：{poc}
    - RSI：{rsi}
    - 主力行為：{action}
    - 系統計算盈虧比：{rr_ratio:.2f}

    【分析任務】
    請依照以下格式，提供不超過 150 字的繁體中文分析：
    1. **趨勢定調**：判斷目前趨勢（多頭、空頭或盤整）。
    2. **具體買入建議**：給出區間（參考 POC）。
    3. **目標與止損**：給出具體點位。
    4. **盈虧比評價**：針對 {rr_ratio:.2f} 的盈虧比給出操作評價。

    直接給結論，不需前言，口吻要果斷專業。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位精通籌碼分佈與技術分析的專業交易員。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"AI 暫時無法連線: {str(e)[:50]}"

async def analyze_stock(ticker):
    """核心分析函數"""
    try:
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 35:
            return f"❌ 無法獲取 {ticker} 的數據，請檢查代碼是否正確。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 1. 技術指標計算
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
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

        # 2. 提取數值
        latest_p = float(df['Close'].iloc[-1])
        pct_chg = ((latest_p / float(df['Close'].iloc[-2])) - 1) * 100
        vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].tail(20).mean())
        cur_rsi = float(df['RSI'].iloc[-1])
        
        # 3. 盈虧比 (Risk/Reward) 計算
        recent_high = float(df['Close'].tail(60).max())
        stop_loss = poc_price * 0.97
        
        risk = latest_p - stop_loss
        reward = recent_high - latest_p
        
        # 防止除以零或負風險
        rr_ratio = reward / risk if risk > 0 else 0
        rr_status = "✅ 盈虧比優良" if rr_ratio >= 1.5 else "⚠️ 盈虧比不具吸引力"
        
        # 4. 判定邏輯
        main_action = "🔥 主力放量進場" if vol_ratio >= VOL_THRESHOLD and pct_chg > 1.8 else \
                      "😱 主力放量派發" if vol_ratio >= VOL_THRESHOLD and pct_chg < -1.8 else "🔘 籌碼縮量整理"
        
        if latest_p > poc_price:
            advice = "多頭結構" if cur_rsi < 68 else "漲幅過快"
        else:
            advice = "弱勢壓制" if cur_rsi > 35 else "超跌區域"

        # 5. 獲取 AI 點評
        ai_note = get_ai_comment(ticker, latest_p, cur_rsi, poc_price, main_action, advice, rr_ratio)

        # 6. 組裝報告
        curr = "HK$ " if ".HK" in ticker else "$ "
        report = (
            f"📊 **{ticker} 深度診斷報告**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 **現價**：{curr}{latest_p:.2f} ({pct_chg:+.2f}%)\n"
            f"📍 **籌碼重心**：{curr}{poc_price:.2f}\n"
            f"⚖️ **盈虧比分析**：{rr_ratio:.2f} ({rr_status})\n"
            f"🔍 **主力行為**：{main_action}\n\n"
            f"🧠 **DeepSeek AI 點評**：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 7. 繪圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label='Price')
        ax1.plot(plot_df.index, plot_df['Upper'], 'g--', alpha=0.3)
        ax1.plot(plot_df.index, plot_df['Lower'], 'r--', alpha=0.3)
        ax1.fill_between(plot_df.index, plot_df['Lower'], plot_df['Upper'], color='gray', alpha=0.1)
        ax1.axhline(poc_price, color='orange', ls='-', alpha=0.8, label=f'POC: {poc_price:.2f}')
        ax1.set_title(f"{ticker} Analysis")
        ax1.legend(loc='upper left')

        ax2.fill_between(plot_df.index, plot_df['Cum_MF'].tail(100), color='purple', alpha=0.1)
        ax2.plot(plot_df.index, plot_df['Cum_MF'].tail(100), color='purple', label='Money Flow')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        return f"❌ 分析出錯: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp_msg = await update.message.reply_text(f"🔍 正在診斷 {ticker} 並計算盈虧比...")
        report, chart = await analyze_stock(ticker)
        if chart:
            await update.message.reply_photo(photo=chart, caption=report, parse_mode='Markdown')
        else:
            await update.message.reply_text(report)
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 您好！我是您的 AI 籌碼助手。傳送代號即可開始深度分析。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 找不到 TELEGRAM_TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 AI 診斷機器人已在 Railway 啟動...")
        app.run_polling(drop_pending_updates=True)
