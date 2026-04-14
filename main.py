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

def get_latest_news(ticker_str, name, t_obj):
    """
    綜合新聞搜尋：優先使用 yfinance 官方新聞，備援使用 DuckDuckGo
    """
    news_results = []
    
    # 方案 A: 使用 yfinance 官方新聞接口 (最穩定，但多為英文)
    try:
        yf_news = t_obj.news
        if yf_news:
            for item in yf_news[:3]:
                news_results.append(f"- {item['title']} (來源: {item['publisher']})")
    except Exception as e:
        logger.warning(f"yfinance news failed: {e}")

    # 方案 B: 如果 yfinance 沒消息或你想加強中文消息，使用 DDG
    if len(news_results) < 2:
        query = f"{name} {ticker_str} 股價 新聞"
        try:
            # 增加隨機性並減少過度頻繁搜尋
            with DDGS(timeout=10) as ddgs:
                results = ddgs.news(query, region="wt-wt", safesearch="off", timelimit="w", max_results=3)
                if results:
                    for r in results:
                        news_results.append(f"- {r['title']} ({r['date']})")
        except Exception as e:
            logger.error(f"DDG news failed: {e}")

    return "\n".join(news_results) if news_results else "暫無近期重大市場消息或搜尋受限。"

def get_ai_comment(ticker, name, data_summary, market_news):
    if not DEEPSEEK_KEY: return "⚠️ AI 密鑰未設定"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"""
    作為首席量化分析師，針對 {name} ({ticker}) 診斷：
    【技術數據摘要】: {data_summary}
    【市場參考消息】: {market_news}
    
    【任務】: 提供 350 字繁體中文報告。
    1. 📰 消息面分析：若消息為英文請翻譯並評估對股價影響；若無具體消息則分析大盤環境。
    2. 📊 籌碼與技術面：分析 POC 重心與 MACD/RSI 背離。
    3. 🛡️ 具體策略：給出進場位與 ATR 止損。
    4. 📈 評分 1-10。
    """
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "你是一位專業對沖基金經理。"}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }, timeout=45)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ AI 診斷暫時失敗 (原因: {str(e)[:30]})"

async def analyze_stock(ticker_str):
    try:
        t_obj = yf.Ticker(ticker_str)
        # 獲取中文名稱
        stock_name = STOCK_NAME_MAP.get(ticker_str)
        if not stock_name:
            try:
                # 備援抓取名稱
                stock_name = t_obj.info.get('shortName') or ticker_str
            except:
                stock_name = ticker_str
        
        # 1. 改進的新聞抓取
        market_news = get_latest_news(ticker_str, stock_name, t_obj)
        
        # 2. 獲取股價數據
        df = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        if df.empty or len(df) < 20:
            return f"❌ 數據量不足: {ticker_str}", None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. 指標計算 (修正 NaN)
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        exp1, exp2 = df['Close'].ewm(span=12).mean(), df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9).mean()
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

        # 6. 繪圖
        plt.style.use('dark_background')
        fig, ax1 = plt.subplots(figsize=(10, 6))
        df.tail(100)['Close'].plot(ax=ax1, color='#00ff00', lw=1.5)
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{stock_name} Technical Analysis")
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        logger.error(f"Major Error: {e}")
        return f"❌ 分析失敗: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp = await update.message.reply_text(f"🔍 正在獲取市場消息與技術診斷: {ticker}...")
        try:
            report, chart = await analyze_stock(ticker)
            if chart:
                try:
                    await update.message.reply_photo(photo=chart, caption=report, parse_mode=ParseMode.MARKDOWN)
                except:
                    await update.message.reply_photo(photo=chart, caption=report)
            else:
                await update.message.reply_text(report)
        finally:
            await temp.delete()

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling(drop_pending_updates=True)
