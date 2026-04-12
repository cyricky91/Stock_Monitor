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

BINS_COUNT = 100 # 增加頻次提高 POC 準確度
VOL_THRESHOLD = 2.0

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def get_ai_comment(ticker, data_summary):
    """呼叫 DeepSeek API 生成對沖基金級別策略點評"""
    if not DEEPSEEK_KEY: return "⚠️ AI 分析未啟用"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"}
    
    prompt = f"""
    作為量化資產管理主管，請針對 {ticker} 進行穿透式診斷：
    
    【量化因子報告】
    {data_summary}
    
    【分析任務：250-300字繁體中文】
    1. 籌碼結構：分析 POC 與現價關係，判斷是「套牢盤沉重」還是「無壓力區」。
    2. 動能與背離：根據 MACD 與 RSI 判斷當前走勢的「真實強度」。
    3. 實戰博弈方案：給出針對性的進場區間、止盈分組與基於 ATR 的動態止損邏輯。
    4. 確定性評分：1-10分。

    要求：冷酷專業，避開空話，直擊股價波動本質。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位專注於風險控制與動量交易的資深基金經理。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3 # 降低隨機性，增加專業度
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"AI 暫時無法連線: {str(e)[:50]}"

async def analyze_stock(ticker):
    """巔峰分析函數：整合多週期均線、MACD、ATR、量價背離"""
    try:
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df_raw.empty or len(df_raw) < 60:
            return f"❌ 數據量不足，無法分析 {ticker}。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # --- 計算技術指標 ---
        # 1. 均線與 MACD
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        # 2. ATR 與 RSI
        tr = pd.concat([df['High']-df['Low'], np.abs(df['High']-df['Close'].shift()), np.abs(df['Low']-df['Close'].shift())], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 3. 籌碼重心 POC (優化)
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # --- 數據抓取 ---
        last = df.iloc[-1]
        latest_p = float(last['Close'])
        cur_rsi = float(last['RSI'])
        atr_v = float(last['ATR'])
        vol_ratio = float(last['Volume'] / df['Volume'].tail(20).mean())
        
        resis = float(df['High'].tail(60).max())
        suppo = float(df['Low'].tail(60).min())
        
        # 背離偵測 (簡單邏輯)
        divergence = "⚠️ 偵測到量價背離" if latest_p > df['Close'].iloc[-5] and vol_ratio < 0.8 else "✅ 量價配合尚可"
        
        # --- AI 摘要 ---
        data_summary = (
            f"- 價格: {latest_p:.2f} (MA50: {last['MA50']:.2f}, MA200: {last['MA200']:.2f})\n"
            f"- 籌碼位: POC {poc_p:.2f}, 60日高低 [{suppo:.2f}-{resis:.2f}]\n"
            f"- 指標: RSI {cur_rsi:.1f}, MACD {last['MACD']:.3f}, ATR {atr_v:.2f}\n"
            f"- 成交倍數: {vol_ratio:.2f}x, {divergence}"
        )

        ai_note = get_ai_comment(ticker, data_summary)

        # --- 格式化報告 ---
        curr = "HK$ " if ".HK" in ticker else "$ "
        trend = "🟢 多頭" if latest_p > last['MA50'] else "🔴 弱勢"
        report = (
            f"👑 **{ticker} 專家級分析報告**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 **現價**：{curr}{latest_p:.2f} ({trend})\n"
            f"📊 **籌碼核心**：{curr}{poc_p:.2f}\n"
            f"🛡️ **ATR 動態止損**：{curr}{latest_p - 2.5*atr_v:.2f}\n"
            f"🚀 **第一目標位**：{curr}{resis:.2f}\n\n"
            f"🧠 **AI 策略點評**：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # --- 圖表繪製 ---
        plt.style.use('ggplot')
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2.5, 0.7, 1]})
        plot_df = df.tail(100)
        
        # P1: 價格與支撐壓力
        ax1.plot(plot_df.index, plot_df['Close'], color='#2c3e50', lw=2, label='Price')
        ax1.plot(plot_df.index, plot_df['MA50'], color='#3498db', lw=1, label='MA50')
        ax1.axhline(poc_p, color='#f39c12', ls='--', alpha=0.8, label=f'POC: {poc_p:.2f}')
        ax1.fill_between(plot_df.index, plot_df['MA50'], plot_df['MA200'].tail(100), color='#ecf0f1', alpha=0.5)
        ax1.set_title(f"{ticker} Technical Framework", fontsize=14)
        ax1.legend(loc='upper left')

        # P2: 成交量與背離偵測
        colors = ['#e74c3c' if c < o else '#2ecc71' for c, o in zip(plot_df['Close'], plot_df['Open'])]
        ax2.bar(plot_df.index, plot_df['Volume'], color=colors, alpha=0.6)
        ax2.axes.get_xaxis().set_visible(False)

        # P3: MACD 與 RSI (雙指標混繪)
        ax3.bar(plot_df.index, plot_df['Hist'], color='gray', alpha=0.2, label='MACD Hist')
        ax3.plot(plot_df.index, plot_df['MACD'], color='#2980b9', lw=1.2, label='MACD')
        ax3.plot(plot_df.index, plot_df['Signal'], color='#e67e22', lw=1.2, label='Signal')
        ax3_rsi = ax3.twinx()
        ax3_rsi.plot(plot_df.index, plot_df['RSI'], color='#8e44ad', lw=0.8, alpha=0.4, label='RSI')
        ax3_rsi.axhline(70, color='r', ls=':', alpha=0.2)
        ax3_rsi.axhline(30, color='g', ls=':', alpha=0.2)
        ax3.legend(loc='upper left', fontsize='small')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=160)
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
        temp = await update.message.reply_text(f"🔋 旗艦級 AI 分析模組啟動：{ticker}...")
        report, chart = await analyze_stock(ticker)
        if chart:
            await update.message.reply_photo(photo=chart, caption=report, parse_mode='Markdown')
        else:
            await update.message.reply_text(report)
        await temp.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💹 指令確認：旗艦級 AI 交易診斷系統已就緒。\n請傳送股票代號（如 700 或 TSLA）。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 找不到 TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🚀 巔峰版機器人已啟動...")
        app.run_polling(drop_pending_updates=True)
