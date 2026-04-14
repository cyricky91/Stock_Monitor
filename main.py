import logging
import re
import io
import os         # 新增：用於讀取環境變數
import requests   # 新增：用於發送 API 請求
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # 新增：防止伺服器繪圖衝突
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# 1. 基礎設定
TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

BINS_COUNT = 100

# 設置日誌
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ai_comment(ticker, data_summary):
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用：請在 Railway 設置 DEEPSEEK_API_KEY。"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"作為量化主管，分析 {ticker}：\n{data_summary}\n請提供 250字繁體中文診斷、ATR止損位與1-10分評分。"
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "你是一位專業策略師。"}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"AI API Error: {e}")
        return f"⚠️ AI 診斷連線失敗: {str(e)[:30]}"

async def analyze_stock(ticker):
    try:
        # 下載數據
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 20:
            return f"❌ 找不到股票代號 {ticker} 或數據不足。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 指標計算
        df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14, min_periods=1).mean()
        
        # 關鍵修正：移除計算產生的空值，防止繪圖報錯
        df = df.dropna(subset=['Close', 'MACD', 'ATR'])

        # POC 籌碼重心
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        data_summary = f"現價:{last['Close']:.2f}, POC:{poc_p:.2f}, MACD:{last['MACD']:.3f}, ATR:{last['ATR']:.2f}"
        ai_note = get_ai_comment(ticker, data_summary)

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
        ax2.bar(plot_df.index, plot_df['Hist'], color='gray', alpha=0.3)
        ax2.plot(plot_df.index, plot_df['MACD'], label='MACD')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return f"❌ 數據處理出錯: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    # 支援逗號或空格分割多個代號
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp_msg = await update.message.reply_text(f"🔋 正在分析 {ticker}...")
        report, chart = await analyze_stock(ticker)
        try:
            if chart:
                await update.message.reply_photo(photo=chart, caption=report, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(report)
        except:
            await update.message.reply_text(report) # 降級處理 Markdown 報錯
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 系統已就緒。傳送股票代號（如 700 或 AAPL）獲取旗艦診斷。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 未設置 TELEGRAM_TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 機器人正在運行中...")
        app.run_polling(drop_pending_updates=True)
