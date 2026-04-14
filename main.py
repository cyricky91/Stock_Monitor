import logging
import re
import io
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS  # 新增：用於免費搜尋新聞
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# 1. 基礎設定
TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
BINS_COUNT = 100

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 繁體中文映射表
STOCK_NAME_MAP = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴", "3690.HK": "美團",
    "1810.HK": "小米集團", "6613.HK": "百威亞太", "3317.HK": "實力建業"
}

def get_latest_news(ticker, name):
    """搜尋該股票最新的 3 則相關新聞"""
    query = f"{name} {ticker} 股價 新聞"
    news_results = []
    try:
        with DDGS() as ddgs:
            # 搜尋最近一週的新聞
            results = ddgs.news(query, region="wt-wt", safesearch="off", timelimit="w", max_results=3)
            for r in results:
                news_results.append(f"- {r['title']} ({r['date']})")
        return "\n".join(news_results) if news_results else "暫無近期重大市場消息。"
    except Exception as e:
        logger.error(f"新聞搜尋失敗: {e}")
        return "新聞搜尋暫時不可用。"

def get_ai_comment(ticker, name, data_summary, market_news):
    """呼叫 AI，將技術數據與新聞結合分析"""
    if not DEEPSEEK_KEY: return "⚠️ AI 密鑰未設定"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"""
    作為資深分析師，針對 {name} ({ticker}) 進行診斷：
    【技術數據】: {data_summary}
    【最新市場新聞】: 
    {market_news}

    【任務】: 提供 350 字左右繁體中文報告。必須包括：
    1. 📰 市場動向：總結新聞對股價是正面還是負面（如收購、業績、政策影響）。
    2. 📊 籌碼與技術面：POC、MACD 與 RSI 分析。
    3. 🛡️ 風控建議：給出具體進場與 ATR 止損價位。
    4. ⚖️ 綜合評分：1-10分。
    """
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "你是一位結合基本面新聞與技術面量化的專家。"}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }, timeout=30)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ AI 分析異常: {str(e)[:50]}"

async def analyze_stock(ticker_str):
    try:
        t_obj = yf.Ticker(ticker_str)
        # 獲取名稱邏輯
        stock_name = STOCK_NAME_MAP.get(ticker_str) or t_obj.info.get('shortName') or ticker_str
        
        # 1. 獲取市場消息 (新增)
        market_news = get_latest_news(ticker_str, stock_name)
        
        # 2. 下載與計算數據
        df = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 60: return f"❌ 無法獲取數據", None
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

        # 指標計算 (簡化版)
        df['MA50'] = df['Close'].rolling(50).mean()
        exp1, exp2 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        # POC 籌碼重心
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        data_summary = f"現價:{last['Close']:.2f}, POC:{poc_p:.2f}, MACD:{last['MACD']:.3f}, ATR:{last['ATR']:.2f}"
        
        # 3. 呼叫 AI 整合新聞與技術面
        ai_note = get_ai_comment(ticker_str, stock_name, data_summary, market_news)

        curr = "HK$ " if ".HK" in ticker_str else "$ "
        report = (
            f"🚀 *{stock_name}* ({ticker_str})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *現價*：{curr}{last['Close']:.2f}\n"
            f"📍 *籌碼重心*：{curr}{poc_p:.2f}\n"
            f"🛡️ *ATR 止損*：{curr}{last['Close'] - 2.5*last['ATR']:.2f}\n\n"
            f"{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 4. 繪圖
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(10, 6))
        df.tail(100)['Close'].plot(ax=ax1, color='#00ff00', title=f"{stock_name} Analysis")
        ax1.axhline(poc_p, color='orange', ls='--')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close(fig)
        return report, buf
    except Exception as e:
        return f"❌ 系統錯誤: {e}", None

# --- Telegram 處理邏輯 (保持不變) ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    for ticker in tickers:
        temp = await update.message.reply_text(f"🔍 正在進行全網搜尋與診斷: {ticker}...")
        report, chart = await analyze_stock(ticker)
        try:
            if chart:
                await update.message.reply_photo(photo=chart, caption=report, parse_mode=ParseMode.MARKDOWN)
            else:
                await update.message.reply_text(report)
        except:
            await update.message.reply_photo(photo=chart, caption=report)
        await temp.delete()

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)
