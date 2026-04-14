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

# 設置日誌
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 繁體中文映射表 ---
STOCK_NAME_MAP = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴", "3690.HK": "美團",
    "1810.HK": "小米集團", "0005.HK": "匯豐控股", "0388.HK": "香港交易所",
    "6613.HK": "百威亞太", "3317.HK": "實力建業", "NVDA": "輝達 (Nvidia)",
    "TSLA": "特斯拉 (Tesla)", "AAPL": "蘋果 (Apple)"
}

def get_latest_news(ticker, name):
    """搜尋該股票最新的市場消息"""
    query = f"{name} {ticker} 股價 新聞 消息"
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

def get_stock_name(ticker_str, t_obj):
    """獲取中文名稱邏輯"""
    if ticker_str in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[ticker_str]
    
    # 嘗試從 Yahoo Search API 獲取名稱
    try:
        search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker_str}&quotesCount=1"
        resp = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        data = resp.json()
        if data.get('quotes'):
            name = data['quotes'][0].get('shortname') or data['quotes'][0].get('longname')
            if name:
                STOCK_NAME_MAP[ticker_str] = name
                return name
    except:
        pass
    
    try:
        return t_obj.info.get('shortName') or ticker_str
    except:
        return ticker_str

def get_ai_comment(ticker, name, data_summary, market_news):
    """呼叫 AI 整合技術數據與市場消息"""
    if not DEEPSEEK_KEY: return "⚠️ AI 密鑰未設定"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"""
    作為首席量化分析師，針對 {name} ({ticker}) 診斷：
    【技術數據摘要】: {data_summary}
    【最新市場動態】: 
    {market_news}

    【任務】: 提供約 350 字繁體中文報告，必須包含：
    1. 📰 市場消息評估：分析新聞對股價是正面還是負面（如收購、業績、傳聞）。
    2. 📊 籌碼與動能分析：解讀 POC、RSI 與 MACD。
    3. 🛡️ 具體策略：基於 ATR 提供進場位與止損位。
    4. 📈 綜合評分：1-10分。
    """
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [{"role": "system", "content": "你是一位專業對沖基金經理，善於結合消息面與技術面。"}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }, timeout=30)
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"⚠️ AI 診斷超時或失敗: {str(e)[:50]}"

async def analyze_stock(ticker_str):
    try:
        t_obj = yf.Ticker(ticker_str)
        stock_name = get_stock_name(ticker_str, t_obj)
        
        # 1. 抓取消息
        market_news = get_latest_news(ticker_str, stock_name)
        
        # 2. 下載數據並修正 NaN 問題
        df = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        if df is None or df.empty or len(df) < 20:
            return f"❌ 數據量不足：{stock_name} ({ticker_str}) 可能近期停牌或無數據。", None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 3. 技術指標計算 (加入 min_periods 防止大量 NaN)
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
        
        # 關鍵：剔除無法繪圖的無效行
        df = df.dropna(subset=['Close', 'MACD', 'ATR'])
        
        if df.empty:
            return f"❌ 指標計算後無有效數據。", None

        # 4. 籌碼分析
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, bins=BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        data_summary = f"現價:{last['Close']:.2f}, POC:{poc_p:.2f}, MACD:{last['MACD']:.3f}, ATR:{last['ATR']:.2f}"
        
        # 5. 獲取 AI 綜合評語
        ai_note = get_ai_comment(ticker_str, stock_name, data_summary, market_news)

        curr = "HK$ " if ".HK" in ticker_str else "$ "
        report = (
            f"🚀 *{stock_name}* ({ticker_str})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *現價*：{curr}{last['Close']:.2f}\n"
            f"📍 *籌碼重心*：{curr}{poc_p:.2f}\n"
            f"🛡️ *ATR 止損位*：{curr}{last['Close'] - 2.5*last['ATR']:.2f}\n\n"
            f"{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 6. 繪圖 (強化穩定性)
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='#00ff00', lw=1.5, label='Price')
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{stock_name} Analysis")
        ax1.legend()
        
        ax2.bar(plot_df.index, plot_df['Hist'], color='cyan', alpha=0.3)
        ax2.plot(plot_df.index, plot_df['MACD'], color='blue', label='MACD')
        ax2.legend()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, facecolor='#1a1a1a')
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return f"❌ 系統錯誤: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp = await update.message.reply_text(f"🔍 正在搜尋市場新聞並診斷: {ticker}...")
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
    await update.message.reply_text("💹 旗艦級消息+量化診斷系統已就緒。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 錯誤: TOKEN MISSING")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🚀 啟動中...")
        app.run_polling(drop_pending_updates=True)
