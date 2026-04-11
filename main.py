import logging
import re
import io
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# 建議將 Token 設定在 Railway 的 Variables 中，這裡改用環境變數讀取
TOKEN = os.getenv("TELEGRAM_TOKEN", "8786623670:AAHRQIvKKX6Gidc9pyqKgl3s17bI7ibk0tU")

# 分析參數
BINS_COUNT = 80
VOL_THRESHOLD = 2.0

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def analyze_stock(ticker):
    try:
        # 下載數據
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 35:
            return f"❌ 無法獲取 {ticker} 的足夠數據", None

        # 重要修正：處理 MultiIndex 結構，確保只抓取該 ticker 的數據
        if isinstance(df_raw.columns, pd.MultiIndex):
            df = pd.DataFrame({
                'Close': df_raw['Close'][ticker],
                'Volume': df_raw['Volume'][ticker]
            }).dropna()
        else:
            df = df_raw[['Close', 'Volume']].dropna()

        # 1. 籌碼重心 (POC)
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, bins=BINS_COUNT, weights=vols)
        poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # 2. RSI 指標
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 3. 累計資金流 (CMF)
        df['MF'] = np.where(df['Close'].diff() > 0, df['Volume'], np.where(df['Close'].diff() < 0, -df['Volume'], 0))
        df['Cum_MF'] = df['MF'].cumsum()

        # 數據提取 - 使用 .iloc[-1] 並轉型，確保是純數值
        latest_p = float(df['Close'].iloc[-1])
        prev_p = float(df['Close'].iloc[-2])
        pct_chg = ((latest_p / prev_p) - 1) * 100
        vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].tail(20).mean())
        cur_rsi = float(df['RSI'].iloc[-1])
        
        # 市場與格式設定
        is_hk = ".HK" in ticker
        curr = "HK$ " if is_hk else "$ "
        
        # 策略邏輯
        main_action = "🔥 主力放量進場" if vol_ratio >= VOL_THRESHOLD and pct_chg > 1.8 else \
                      "😱 主力放量派發" if vol_ratio >= VOL_THRESHOLD and pct_chg < -1.8 else "🔘 籌碼縮量整理"
        
        advice = "📈 多頭結構，建議分批佈局" if latest_p > poc_price and cur_rsi < 68 else \
                 "⚠️ 漲幅過快，暫不追高" if latest_p > poc_price else \
                 "📉 弱勢壓制，建議觀望" if cur_rsi > 35 else "🔵 超跌區域，靜待反彈"

        report = (
            f"📊 **{ticker} 深度分析報告**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 **現價**：{curr}{latest_p:.2f} ({pct_chg:+.2f}%)\n"
            f"📍 **籌碼重心**：{curr}{poc_price:.2f}\n"
            f"🔍 **主力行為**：{main_action}\n"
            f"💡 **交易建議**：{advice}\n\n"
            f"🎯 **操作參考**：\n"
            f"  - 買入支撐位：{curr}{poc_price:.2f}\n"
            f"  - 止損參考位：{curr}{poc_price * 0.95:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 繪圖
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(df.index[-100:], df['Close'].tail(100), color='#1f77b4', lw=2)
        ax1.axhline(poc_price, color='red', ls='--', alpha=0.7)
        ax2.fill_between(df.index[-100:], df['Cum_MF'].tail(100), color='purple', alpha=0.1)
        ax2.plot(df.index[-100:], df['Cum_MF'].tail(100), color='purple')
        plt.tight_layout()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
        except Exception as e:
        return f"❌ 分析出錯: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.upper().strip()
    # 自動補全：純數字 -> 港股, 其他 -> 美股
    tickers = [s.zfill(4)+".HK" if s.isdigit() else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        msg = await update.message.reply_text(f"🔍 正在診斷 {ticker}...")
        report, chart = await analyze_stock(ticker)
        if chart:
            await update.message.reply_photo(photo=chart, caption=report, parse_mode='Markdown')
        else:
            await update.message.reply_text(report)
        await msg.delete() # 刪除「正在診斷」的提示訊息

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 你好！直接傳送股票代號（例如：0005 或 NVDA），我會立刻為你分析籌碼分佈。")

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 即時診斷機器人已在 Railway 啟動...")
    app.run_polling()
