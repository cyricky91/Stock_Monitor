import logging
import re
import io
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

def get_ai_comment(ticker, data_summary, info_summary, news_summary):
    """呼叫 AI，要求包含背景、競爭、新聞分析及精確點位"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用：請設置 API Key。"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"""
    作為資深首席量化投資官，針對股票 {ticker} 進行全方位深度分析：
    
    【基本面資訊】: {info_summary}
    【最新新聞消息】: {news_summary}
    【技術面數據】: {data_summary}
    
    【任務】請提供約 500 字的繁體中文報告，嚴格包含以下結構：
    1. 🏢 股票背景與同業競爭：簡述公司業務定位及主要對手。
    2. 📰 新聞消息解讀：分析最近新聞對股價的影響（利好、利空或中性），以及市場情緒。
    3. 📊 技術診斷：分析籌碼重心(POC)、均線斜率與 MACD 動能。
    4. 🎯 具體交易方案：
       - 進場點：建議具體價格。
       - 加倉點：後續補倉位置。
       - 止盈點：近期壓力位。
       - 止蝕點：防守支撐位。
    5. ⭐ 綜合投資評分 (1-10)。
    """
    
    try:
        response = requests.post(url, headers=headers, json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一位精通全球基本面與技術面的對沖基金經理，善於從新聞中洞察市場預期。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }, timeout=45) # 增加超時時間，因為新聞處理較久
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        logger.error(f"AI API Error: {e}")
        return f"⚠️ AI 診斷連線失敗: {str(e)[:50]}"

async def analyze_stock(ticker):
    try:
        t_obj = yf.Ticker(ticker)
        
        # 1. 提取基本面資料
        info = t_obj.info
        company_name = info.get('longName', ticker)
        industry = info.get('industry', '未知行業')
        sector = info.get('sector', '未知板塊')
        summary = info.get('longBusinessSummary', '暫無簡介')[:250]
        info_summary = f"公司名稱: {company_name}, 行業: {sector}/{industry}, 業務概要: {summary}"
        
        # 2. 獲取最新新聞
        news = t_obj.news[:5] # 獲取最近5則
        news_list = []
        for n in news:
            title = n.get('title', '')
            publisher = n.get('publisher', '')
            news_list.append(f"[{publisher}] {title}")
        news_summary = " | ".join(news_list) if news_list else "近期無重大公開新聞消息。"

        # 3. 下載價格數據
        df_raw = t_obj.history(period="1y", interval="1d", auto_adjust=True)
        if df_raw.empty or len(df_raw) < 20:
            return f"❌ 數據下載失敗：{ticker} 數據不足。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 指標計算
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14, min_periods=1).mean()
        df = df.dropna(subset=['Close', 'MACD', 'ATR'])

        # POC 籌碼重心
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        last = df.iloc[-1]
        data_summary = f"現價:{last['Close']:.2f}, POC:{poc_p:.2f}, MACD:{last['MACD']:.3f}, ATR:{last['ATR']:.2f}"
        
        # AI 深度分析 (加入新聞參數)
        ai_note = get_ai_comment(ticker, data_summary, info_summary, news_summary)

        # 報告組裝
        curr = "HK$ " if ".HK" in ticker else "$ "
        report = (
            f"🚀 *{company_name}* ({ticker})\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *現價*：{curr}{last['Close']:.2f}\n"
            f"📍 *籌碼重心*：{curr}{poc_p:.2f}\n"
            f"🛡️ *ATR 止損參考*：{curr}{last['Close'] - 2.5*last['ATR']:.2f}\n\n"
            f"{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 繪圖
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='black', lw=1.5, label='Price')
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{company_name} ({ticker}) Analysis")
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

# (其餘 handle_message 與 main 函數保持不變)
