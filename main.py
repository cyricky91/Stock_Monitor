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
from telegram.constants import ParseMode

# 1. 基礎設定
TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

BINS_COUNT = 100
VOL_THRESHOLD = 2.0

# 設置日誌
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_ai_comment(ticker, name, data_summary):
    """呼叫 DeepSeek API，增加股票名稱以便 AI 理解背景"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用：請在 Railway 設置 DEEPSEEK_API_KEY。"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"""
    作為量化資產管理主管，請針對 {name} ({ticker}) 進行診斷：
    【數據包】: {data_summary}
    【任務】: 提供 250-300 字繁體中文分析。包括：
    1. 籌碼結構與 POC 意義。
    2. 動能與 RSI/MACD 背離偵測。
    3. 基於 ATR 的進場與防守方案（提供具體價位）。
    4. 1-10分評分。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位頂級對沖基金經理，說話冷靜、數據驅動。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"AI API Error: {str(e)}")
        return f"⚠️ AI 診斷連線失敗。原因: {str(e)[:50]}"

async def analyze_stock(ticker_str):
    """核心分析函數：增加股票名稱獲取"""
    try:
        # 獲取 Ticker 對象以獲取名稱
        t_obj = yf.Ticker(ticker_str)
        # 嘗試獲取名稱，若無則用代號代替
        stock_name = t_obj.info.get('shortName') or t_obj.info.get('longName') or ticker_str
        
        # 下載數據
        df_raw = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 60:
            return f"❌ 數據下載失敗：找不到股票 {ticker_str}。", None

        df = df_raw.copy()
        # 處理 MultiIndex 欄位
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # --- 技術指標計算 ---
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # --- 數據摘要 ---
        last = df.iloc[-1]
        latest_p = float(last['Close'])
        atr_v = float(last['ATR'])
        resis = float(df['High'].tail(60).max())
        
        data_summary = (
            f"現價:{latest_p:.2f}, POC:{poc_p:.2f}, RSI:{last['RSI']:.1f}, "
            f"MACD:{last['MACD']:.3f}, ATR:{atr_v:.2f}, MA50:{last['MA50']:.2f}"
        )

        ai_note = get_ai_comment(ticker_str, stock_name, data_summary)

        # --- 報告組裝 ---
        curr = "HK$ " if ".HK" in ticker_str else "$ "
        report = (
            f"🚀 *{stock_name}* ({ticker_str})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *現價*：{curr}{latest_p:.2f}\n"
            f"📍 *籌碼重心*：{curr}{poc_p:.2f}\n"
            f"🛡️ *ATR 止損位*：{curr}{latest_p - 2.5*atr_v:.2f}\n"
            f"🎯 *60日壓力*：{curr}{resis:.2f}\n\n"
            f"🧠 *AI 策略分析*：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # --- 繪圖 ---
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='black', lw=1.5, label='Price')
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{stock_name} Technical Status")
        ax1.legend()

        ax2.bar(plot_df.index, plot_df['Hist'], color='gray', alpha=0.3)
        ax2.plot(plot_df.index, plot_df['MACD'], label='MACD')
        ax2.plot(plot_df.index, plot_df['Signal'], label='Signal')
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
                except Exception:
                    await update.message.reply_photo(photo=chart, caption=report)
            else:
                await update.message.reply_text(report)
        except Exception as final_e:
            logger.error(f"Send Failed: {final_e}")
        
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 系統已就緒。傳送股票代號獲取含名稱的深度診斷。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 錯誤: 未設置 TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 機器人正在運行中...")
        app.run_polling(drop_pending_updates=True)
