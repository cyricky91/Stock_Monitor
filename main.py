import logging
import re
import io
import os         # 修正：補上 os 模組
import requests   # 修正：修復損壞的導入
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # 修正：防止伺服器端繪圖崩潰
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# 1. 基礎設定
TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

BINS_COUNT = 100

# 設置日誌
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_ai_comment(ticker, data_summary):
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用：請在環境變數設置 API Key。"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"作為量化主管，分析 {ticker}：\n{data_summary}\n請提供 250字繁體中文診斷、ATR止損位與1-10分評分。"
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一位精準的基金經理。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"AI API Error: {str(e)}")
        return f"⚠️ AI 診斷連線失敗: {str(e)[:50]}"

async def analyze_stock(ticker):
    try:
        # 下載數據
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 60:
            return f"❌ 數據下載失敗：{ticker} 數據不足。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 指標計算 (min_periods=1 確保不會整段變 NaN)
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
        
        # 核心修正：移除 NaN，防止繪圖崩潰
        df = df.dropna(subset=['Close', 'MACD', 'ATR'])

        # POC 籌碼重心
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        data_summary = f"現價:{last['Close']:.2f}, POC:{poc_p:.2f}, MACD:{last['MACD']:.3f}, ATR:{last['ATR']:.2f}"
        ai_note = get_ai_comment(ticker, data_summary)

        # 報告組裝
        curr = "HK$ " if ".HK" in ticker else "$ "
        report = (
            f"🚀 *{ticker} 診斷報告*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *現價*：{curr}{last['Close']:.2f}\n"
            f"📍 *籌碼重心*：{curr}{poc_p:.2f}\n"
            f"🛡️ *ATR 止損位*：{curr}{last['Close'] - 2.5*last['ATR']:.2f}\n\n"
            f"🧠 *AI 策略分析*：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 繪圖
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='black', lw=1.5, label='Price')
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{ticker} Analysis")
        ax1.legend()

        ax2.bar(plot_df.index, plot_df['Hist'], color='gray', alpha=0.3)
        ax2.plot(plot_df.index, plot_df['MACD'], label='MACD')
        ax2.legend()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        logger.error(f"Analysis Error: {str(e)}")
        return f"❌ 數據處理出錯: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp_msg = await update.message.reply_text(f"🔋 正在分析 {ticker}...")
        report, chart = await analyze_stock(ticker)
        try:
            if chart:
                try:
                    await update.message.reply_photo(photo=chart, caption=report, parse_mode=ParseMode.MARKDOWN)
                except:
                    await update.message.reply_photo(photo=chart, caption=report)
            else:
                await update.message.reply_text(report)
        finally:
            await temp_msg.delete()

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 錯誤: 未設置 TELEGRAM_TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("👋 系統就緒")))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 機器人運行中...")
        app.run_polling(drop_pending_updates=True)
