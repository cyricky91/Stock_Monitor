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

BINS_COUNT = 80
VOL_THRESHOLD = 2.0

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def get_ai_comment(ticker, data_summary):
    """呼叫 DeepSeek API 生成深度點評 (300字內)"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"""
    作為專業資深首席策略師，請針對 {ticker} 進行深度技術診斷：
    
    【核心數據】
    {data_summary}
    
    【分析要求】
    請提供約 200-300 字的繁體中文分析：
    1. 形態分析：根據現價與支撐壓力位，判斷目前處於什麼階段。
    2. 籌碼解讀：結合 POC 與主力行為給出評價。
    3. 操作指南：提供具體的進場區間、目標價位、以及基於 ATR 的防守建議。
    4. 風險提示：若盈虧比不佳或 RSI 過高，請給出警示。
    
    要求：口吻專業果斷，邏輯條理清晰，直接給結論。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位精通量化籌碼與技術派系的頂級交易員。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.6
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"AI 暫時無法連線: {str(e)[:50]}"

async def analyze_stock(ticker):
    """進階分析函數：包含 ATR、支撐壓力與盈虧比"""
    try:
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df_raw.empty or len(df_raw) < 35:
            return f"❌ 無法獲取 {ticker} 的數據。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # --- 技術指標計算 ---
        # 1. 籌碼重心 POC
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # 2. ATR 計算 (14天)
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # 3. RSI & 布林通道
        df['Price_Chg'] = df['Close'].diff()
        gain = (df['Price_Chg'].where(df['Price_Chg'] > 0, 0)).rolling(window=14).mean()
        loss = (-df['Price_Chg'].where(df['Price_Chg'] < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)

        # 4. 提取數值與支撐壓力
        latest_p = float(df['Close'].iloc[-1])
        prev_p = float(df['Close'].iloc[-2])
        atr_v = float(df['ATR'].iloc[-1])
        cur_rsi = float(df['RSI'].iloc[-1])
        
        resistance = float(df['High'].tail(60).max()) # 60日壓力
        support = float(df['Low'].tail(60).min())     # 60日支撐
        
        # --- 邏輯判斷 ---
        pct_chg = ((latest_p / prev_p) - 1) * 100
        vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].tail(20).mean())
        
        main_action = "🔥 主力放量進場" if vol_ratio >= VOL_THRESHOLD and pct_chg > 1.8 else \
                      "😱 主力放量派發" if vol_ratio >= VOL_THRESHOLD and pct_chg < -1.8 else "🔘 籌碼縮量整理"
        
        # 盈虧比計算
        risk = latest_p - (latest_p - 2 * atr_v) # 風險以 2 倍 ATR 計
        reward = resistance - latest_p
        rr_ratio = reward / risk if risk > 0 else 0

        # --- 生成數據摘要給 AI ---
        data_summary = (
            f"- 現價: {latest_p:.2f}\n"
            f"- 漲跌幅: {pct_chg:+.2f}%\n"
            f"- 籌碼重心 (POC): {poc_price:.2f}\n"
            f"- 60日壓力/支撐: {resistance:.2f} / {support:.2f}\n"
            f"- RSI: {cur_rsi:.1f}\n"
            f"- ATR: {atr_v:.2f}\n"
            f"- 主力行為: {main_action}\n"
            f"- 系統計算盈虧比: {rr_ratio:.2f}"
        )

        ai_note = get_ai_comment(ticker, data_summary)

        # --- 組裝報告 ---
        curr = "HK$ " if ".HK" in ticker else "$ "
        report = (
            f"📊 **{ticker} 深度診斷報告**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 **現價**：{curr}{latest_p:.2f} ({pct_chg:+.2f}%)\n"
            f"📍 **籌碼重心**：{curr}{poc_price:.2f}\n"
            f"🛡️ **ATR 止損位**：{curr}{latest_p - 2*atr_v:.2f}\n"
            f"⚡ **60日壓力**：{curr}{resistance:.2f}\n"
            f"⚖️ **盈虧比**：{rr_ratio:.2f}\n\n"
            f"🧠 **深度 AI 策略點評**：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # --- 繪圖 ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='#1f77b4', lw=2, label='Price')
        ax1.plot(plot_df.index, plot_df['Upper'], 'g--', alpha=0.3, label='Bollinger')
        ax1.plot(plot_df.index, plot_df['Lower'], 'r--', alpha=0.3)
        ax1.axhline(poc_price, color='orange', ls='-', alpha=0.8, label=f'POC: {poc_price:.2f}')
        ax1.axhline(resistance, color='red', ls=':', alpha=0.5, label='Resistance')
        ax1.axhline(support, color='green', ls=':', alpha=0.5, label='Support')
        ax1.set_title(f"{ticker} Technical Analysis")
        ax1.legend(loc='upper left', fontsize='small')

        ax2.fill_between(plot_df.index, plot_df['RSI'], 70, where=(plot_df['RSI'] >= 70), color='r', alpha=0.3)
        ax2.fill_between(plot_df.index, plot_df['RSI'], 30, where=(plot_df['RSI'] <= 30), color='g', alpha=0.3)
        ax2.plot(plot_df.index, plot_df['RSI'], color='gray', label='RSI')
        ax2.axhline(70, color='r', ls='--', alpha=0.2)
        ax2.axhline(30, color='g', ls='--', alpha=0.2)
        ax2.legend(loc='upper left')
        
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
        temp_msg = await update.message.reply_text(f"🔍 正在進行進階診斷分析 {ticker}...")
        report, chart = await analyze_stock(ticker)
        if chart:
            await update.message.reply_photo(photo=chart, caption=report, parse_mode='Markdown')
        else:
            await update.message.reply_text(report)
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 您好！傳送股票代號，我將為您提供含 ATR 與支撐壓力的深度分析。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 找不到 TELEGRAM_TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 進階 AI 診斷機器人已啟動...")
        app.run_polling(drop_pending_updates=True)
