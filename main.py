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
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 小型繁體中文映射表 ---
# 你可以根據需求在這裡自由增加常用的股票
STOCK_NAME_MAP = {
    "0700.HK": "騰訊控股",
    "9988.HK": "阿里巴巴",
    "3690.HK": "美團",
    "1810.HK": "小米集團",
    "2318.HK": "中國平安",
    "0005.HK": "匯豐控股",
    "0388.HK": "香港交易所",
    "1211.HK": "比亞迪股份",
    "9618.HK": "京東集團",
    "6613.HK": "百威亞太",
    "3317.HK": "實力建業",
    "NVDA": "輝達 (Nvidia)",
    "TSLA": "特斯拉 (Tesla)",
    "AAPL": "蘋果 (Apple)",
    "MSFT": "微軟 (Microsoft)"
}

def get_stock_name(ticker_str, t_obj):
    """優先從映射表找中文名，否則從 ticker.info 抓取"""
    if ticker_str in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[ticker_str]
    
    try:
        # 嘗試從 API 獲取名稱
        info = t_obj.info
        return info.get('shortName') or info.get('longName') or ticker_str
    except:
        return ticker_str

def get_ai_comment(ticker, name, data_summary):
    if not DEEPSEEK_KEY: return "⚠️ AI 密鑰未設定"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"""
    作為首席策略官，針對 {name} ({ticker}) 診斷：
    【數據】: {data_summary}
    【任務】: 250-300字繁體中文。分析籌碼、MACD背離及ATR止損方案，最後給1-10分。
    """
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "專業對沖基金經理。"}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }, timeout=25)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ AI 分析異常: {str(e)[:50]}"

async def analyze_stock(ticker_str):
    try:
        t_obj = yf.Ticker(ticker_str)
        stock_name = get_stock_name(ticker_str, t_obj) # 獲取中文名稱
        
        df = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 60:
            return f"❌ 無法獲取 {ticker_str} 數據。", None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 指標計算
        df['MA50'] = df['Close'].rolling(50).mean()
        df['MA200'] = df['Close'].rolling(200).mean()
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        latest_p = float(last['Close'])
        atr_v = float(last['ATR'])
        resis = float(df['High'].tail(60).max())
        
        data_summary = f"現價:{latest_p:.2f}, POC:{poc_p:.2f}, RSI:{last['RSI']:.1f}, MACD:{last['MACD']:.3f}, ATR:{atr_v:.2f}"
        ai_note = get_ai_comment(ticker_str, stock_name, data_summary)

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

        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='black', lw=1.5, label='Price')
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{stock_name} ({ticker_str})")
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
        logger.error(f"Error: {e}")
        return f"❌ 處理 {ticker_str} 時出錯: {str(e)}", None

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
        except Exception as e:
            logger.error(f"Send Error: {e}")
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💹 旗艦級 AI 診斷機器人已就緒，請傳送代號。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 未設置 TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 機器人運行中...")
        app.run_polling(drop_pending_updates=True)
