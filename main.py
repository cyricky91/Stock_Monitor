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

# 設置日誌
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 繁體中文映射表 (確保最核心股票秒回中文) ---
STOCK_NAME_MAP = {
    "0700.HK": "騰訊控股",
    "9988.HK": "阿里巴巴",
    "3690.HK": "美團",
    "1810.HK": "小米集團",
    "0005.HK": "匯豐控股",
    "0388.HK": "香港交易所",
    "6613.HK": "百威亞太",
    "3317.HK": "實力建業",
    "NVDA": "輝達 (Nvidia)",
    "TSLA": "特斯拉 (Tesla)",
    "AAPL": "蘋果 (Apple)"
}

def get_hk_name_auto(ticker_str):
    """
    自動抓取港股名稱的備援邏輯
    使用 Yahoo Finance 的 API 提示接口，這比爬取 HKEX 網頁更穩定且快速
    """
    if not ticker_str.endswith(".HK"):
        return None
    
    symbol = ticker_str.replace(".HK", "")
    search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={symbol}&quotesCount=1&newsCount=0"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(search_url, headers=headers, timeout=5)
        data = resp.json()
        if data.get('quotes'):
            return data['quotes'][0].get('shortname')
    except Exception as e:
        logger.error(f"自動抓取名稱失敗: {e}")
    return None

def get_stock_name(ticker_str, t_obj):
    # 優先級 1: 手動映射表
    if ticker_str in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[ticker_str]
    
    # 優先級 2: 自動抓取 (針對港股)
    if ".HK" in ticker_str:
        auto_name = get_hk_name_auto(ticker_str)
        if auto_name:
            STOCK_NAME_MAP[ticker_str] = auto_name # 存入緩存
            return auto_name
            
    # 優先級 3: yfinance 默認名稱
    try:
        return t_obj.info.get('shortName') or ticker_str
    except:
        return ticker_str

def get_ai_comment(ticker, name, data_summary):
    if not DEEPSEEK_KEY: return "⚠️ AI 密鑰未設定"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"作為專業經理人，分析 {name} ({ticker})：\n{data_summary}\n請提供 250字繁體中文診斷、ATR止損建議與1-10評分。"
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "你是精通量化與籌碼分析的專家。"}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }, timeout=25)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ AI 分析超時或失敗: {str(e)[:50]}"

async def analyze_stock(ticker_str):
    try:
        t_obj = yf.Ticker(ticker_str)
        stock_name = get_stock_name(ticker_str, t_obj)
        
        # 下載數據，增加超時處理
        df = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 60:
            return f"❌ 無法獲取 {ticker_str} 的數據。", None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 計算指標
        df['MA50'] = df['Close'].rolling(50).mean()
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        latest_p = float(last['Close'])
        atr_v = float(last['ATR'])
        
        data_summary = f"現價:{latest_p:.2f}, POC:{poc_p:.2f}, MACD:{last['MACD']:.3f}, ATR:{atr_v:.2f}"
        ai_note = get_ai_comment(ticker_str, stock_name, data_summary)

        curr = "HK$ " if ".HK" in ticker_str else "$ "
        report = (
            f"🚀 *{stock_name}* ({ticker_str})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *現價*：{curr}{latest_p:.2f}\n"
            f"📍 *籌碼重心*：{curr}{poc_p:.2f}\n"
            f"🛡️ *ATR 止損位*：{curr}{latest_p - 2.5*atr_v:.2f}\n\n"
            f"🧠 *AI 策略分析*：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 繪圖優化
        plt.style.use('dark_background') # 改為深色模式配合 Telegram
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='#00ff00', lw=1.5, label='Price')
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{stock_name} ({ticker_str})", color='white')
        ax2.bar(plot_df.index, plot_df['Hist'], color='cyan', alpha=0.3)
        ax2.plot(plot_df.index, plot_df['MACD'], color='blue')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, facecolor='#1a1a1a')
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        logger.error(f"分析失敗: {e}")
        return f"❌ 系統錯誤: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp = await update.message.reply_text(f"🔍 正在獲取 {ticker} 的深度數據...")
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
            logger.error(f"發送失敗: {e}")
        await temp.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💹 旗艦診斷系統已就緒。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ TOKEN MISSING")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🚀 啟動成功...")
        app.run_polling(drop_pending_updates=True)
