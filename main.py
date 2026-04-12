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

# 1. 基礎設定
TOKEN = os.getenv("TELEGRAM_TOKEN")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")

BINS_COUNT = 80
VOL_THRESHOLD = 2.0

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def get_ai_comment(ticker, data_summary):
    """呼叫 DeepSeek API 生成深度策略點評"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"""
    作為專業對沖基金經理，請對股票 {ticker} 進行診斷：
    
    【量化數據包】
    {data_summary}
    
    【分析任務：請提供約 250-300 字繁體中文精準分析】
    1. 現狀定性：目前是「主升浪、反彈中、築底期、還是派發末端」。
    2. 動能評估：結合 MACD 與量價關係，判斷目前上漲/下跌是否有力。
    3. 實戰建議：根據 ATR 止損與支撐壓力，給出精確的「進場點位、第一止盈目標、終極防守線」。
    4. 風險評級：(1-5星)，並簡述理由。

    要求：冷靜、精準、具備可操作性。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位頂級量化交易員，擅長識別趨勢轉折與風險回報比分析。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.4
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"AI 暫時無法連線: {str(e)[:50]}"

async def analyze_stock(ticker):
    """旗艦分析函數：整合 MACD, ATR, 量價與均線"""
    try:
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df_raw.empty or len(df_raw) < 60:
            return f"❌ 數據量不足，無法分析 {ticker}。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # --- 計算指標 ---
        # 1. 均線系統
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # 2. MACD 計算
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        # 3. ATR 與 RSI
        tr = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 4. 籌碼重心 POC
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # --- 數據抓取 ---
        last = df.iloc[-1]
        prev = df.iloc[-2]
        latest_p = float(last['Close'])
        cur_rsi = float(last['RSI'])
        atr_v = float(last['ATR'])
        vol_ratio = float(last['Volume'] / df['Volume'].tail(20).mean())
        
        resis = float(df['High'].tail(60).max())
        suppo = float(df['Low'].tail(60).min())
        
        # 狀態標籤
        market_sentiment = "🔥 極度貪婪" if cur_rsi > 75 else "❄️ 恐慌超跌" if cur_rsi < 25 else "⚖️ 情緒平穩"
        macd_status = "📈 金叉向上" if last['MACD'] > last['Signal'] else "📉 死叉向下"
        
        # --- AI 數據摘要 ---
        data_summary = (
            f"- 價格: 現價 {latest_p:.2f}, POC {poc_p:.2f}\n"
            f"- 支撐/壓力: {suppo:.2f} / {resis:.2f}\n"
            f"- 均線位: MA50 {last['MA50']:.2f}, MA200 {last['MA200']:.2f}\n"
            f"- 動能: MACD {macd_status}, RSI {cur_rsi:.1f}\n"
            f"- 波動: ATR {atr_v:.2f}, 成交量倍數 {vol_ratio:.2f}x\n"
            f"- 情緒標籤: {market_sentiment}"
        )

        ai_note = get_ai_comment(ticker, data_summary)

        # --- 報告組裝 ---
        curr = "HK$ " if ".HK" in ticker else "$ "
        report = (
            f"🚀 **{ticker} 旗艦診斷報告**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 **現價**：{curr}{latest_p:.2f}\n"
            f"🎭 **情緒**：{market_sentiment}\n"
            f"📉 **動能**：{macd_status}\n"
            f"🛡️ **ATR 防守位**：{curr}{latest_p - 2*atr_v:.2f}\n\n"
            f"🧠 **AI 首席策略分析**：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # --- 進階圖表繪製 ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 12), gridspec_kw={'height_ratios': [2.5, 0.8, 1]})
        plot_df = df.tail(100)
        
        # P1: 價格與均線
        ax1.plot(plot_df.index, plot_df['Close'], color='black', lw=1.5, label='Price')
        ax1.plot(plot_df.index, plot_df['MA50'], color='blue', alpha=0.6, label='MA50')
        ax1.plot(plot_df.index, plot_df['MA200'], color='purple', alpha=0.6, label='MA200')
        ax1.axhline(poc_p, color='orange', ls='--', alpha=0.7, label='POC')
        ax1.fill_between(plot_df.index, plot_df['MA50'], plot_df['MA200'], color='gray', alpha=0.1)
        ax1.set_title(f"{ticker} Master Analysis")
        ax1.legend(loc='upper left', fontsize='small')

        # P2: 成交量
        ax2.bar(plot_df.index, plot_df['Volume'], color='silver', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.axes.get_xaxis().set_visible(False)

        # P3: MACD
        ax3.bar(plot_df.index, plot_df['Hist'], color='gray', alpha=0.3, label='MACD Hist')
        ax3.plot(plot_df.index, plot_df['MACD'], color='blue', lw=1, label='MACD')
        ax3.plot(plot_df.index, plot_df['Signal'], color='red', lw=1, label='Signal')
        ax3.legend(loc='upper left', fontsize='small')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        return f"❌ 分析出錯: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    tickers = [s.zfill(4)+".HK" if s.isdigit() else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp_msg = await update.message.reply_text(f"🔥 正在調用旗艦級 AI 策略分析 {ticker}...")
        report, chart = await analyze_stock(ticker)
        if chart:
            await update.message.reply_photo(photo=chart, caption=report, parse_mode='Markdown')
        else:
            await update.message.reply_text(report)
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💎 歡迎使用旗艦 AI 投資診斷系統。\n傳送股票代號，為您解碼市場動態。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 找不到 TELEGRAM_TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 旗艦版 AI 機器人已就緒...")
        app.run_polling(drop_pending_updates=True)
