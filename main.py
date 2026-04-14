import logging
import re
import io
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from duckduckgo_search import DDGS
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode

# 1. 基礎設定
TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
BINS_COUNT = 100

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 繁體中文映射表 ---
STOCK_NAME_MAP = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴", "3690.HK": "美團",
    "1810.HK": "小米集團", "0005.HK": "匯豐控股", "0388.HK": "香港交易所",
    "6613.HK": "百威亞太", "3317.HK": "實力建業", "0011.HK": "恒生銀行"
}

def get_latest_news(ticker, name):
    """搜尋該股票最新的市場消息 (增加超時與錯誤處理)"""
    query = f"{name} {ticker} 股價 新聞"
    news_results = []
    try:
        # 限制搜尋時間，避免卡死
        with DDGS(timeout=10) as ddgs:
            results = ddgs.news(query, region="wt-wt", safesearch="off", timelimit="w", max_results=3)
            if results:
                for r in results:
                    news_results.append(f"- {r['title']} ({r['date']})")
        return "\n".join(news_results) if news_results else "暫無近期重大市場消息。"
    except Exception as e:
        logger.error(f"新聞搜尋異常 (跳過新聞): {e}")
        return "新聞搜尋暫時不可用，僅進行技術面分析。"

def get_ai_comment(ticker, name, data_summary, market_news):
    if not DEEPSEEK_KEY: return "⚠️ AI 密鑰未設定"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    # 優化 Prompt 結構
    prompt = f"""
    作為首席量化分析師，針對 {name} ({ticker}) 診斷：
    【技術數據摘要】: {data_summary}
    【市場動態】: {market_news}
    【任務】: 提供 350 字繁體中文報告。必須包括：
    1. 📰 市場評估（新聞影響）。
    2. 📊 籌碼與動能（POC/MACD）。
    3. 🛡️ 具體策略（ATR 止損與進場）。
    4. 📈 評分 1-10。
    """
    
    try:
        # 將超時增加到 45 秒，因為加入新聞後 AI 處理變慢
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "你是一位專業基金經理。"}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }, timeout=45)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"DeepSeek API Error: {e}")
        return f"⚠️ AI 診斷暫時中斷 (原因: {str(e)[:30]})"

async def analyze_stock(ticker_str):
    try:
        t_obj = yf.Ticker(ticker_str)
        # 獲取名稱
        stock_name = STOCK_NAME_MAP.get(ticker_str)
        if not stock_name:
            try:
                info = t_obj.info
                stock_name = info.get('shortName') or ticker_str
            except:
                stock_name = ticker_str
        
        # 1. 抓取消息 (就算失敗也不要讓程式崩潰)
        market_news = get_latest_news(ticker_str, stock_name)
        
        # 2. 下載數據
        df = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 15:
            return f"❌ 數據量不足: {ticker_str}", None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. 指標計算
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        exp1, exp2 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
        
        df = df.dropna(subset=['Close', 'MACD'])
        
        # 4. 籌碼分析
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, bins=BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        data_summary = f"現價:{last['Close']:.2f}, POC:{poc_p:.2f}, MACD:{last['MACD']:.3f}, ATR:{last['ATR']:.2f}"
        
        # 5. AI 分析
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

        # 繪圖
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(10, 6))
        df.tail(100)['Close'].plot(ax=ax1, color='#00ff00', lw=1.5)
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{stock_name} ({ticker_str})")
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        logger.error(f"重大分析錯誤: {e}")
        return f"❌ 分析中斷: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp = await update.message.reply_text(f"🔍 正在搜尋市場新聞並診斷: {ticker}...")
        try:
            report, chart = await analyze_stock(ticker)
            if chart:
                try:
                    await update.message.reply_photo(photo=chart, caption=report, parse_mode=ParseMode.MARKDOWN)
                except:
                    # 如果 Markdown 報錯，嘗試以普通文字發送
                    await update.message.reply_photo(photo=chart, caption=report)
            else:
                await update.message.reply_text(report)
        except Exception as outer_e:
            await update.message.reply_text(f"❌ 系統執行異常: {outer_e}")
        finally:
            # 確保無論成功與否都刪除「正在搜尋」的訊息
            await temp.delete()

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)
