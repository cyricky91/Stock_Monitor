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
    """呼叫 DeepSeek API 生成 300字深度點評"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"""
    作為首席量化策略師，請針對股票 {ticker} 進行深度技術診斷：
    
    【核心技術面數據】
    {data_summary}
    
    【分析任務：請提供約 250-300 字繁體中文評論】
    1. 趨勢定位：判斷長中短期走勢（參考現價與 MA50/MA200 關係）。
    2. 量價解讀：分析目前成交量是否支撐現股價，有無量價背離。
    3. 實戰建議：根據盈虧比與 ATR，給出明確的「進場路徑、分批止盈點、硬性止損位」。
    4. 總結評分：給出 1-10 分的投資吸引力評分。
    
    要求：專業、犀利、數據驅動，不說廢話。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位精通籌碼分佈、動量策略與風險管理的資深交易主管。"},
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
    """終極分析函數：包含 MA、ATR、支撐壓力、量價分析"""
    try:
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df_raw.empty or len(df_raw) < 50:
            return f"❌ 數據量不足，無法分析 {ticker}。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # --- 計算技術指標 ---
        # 1. 均線系統
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # 2. 籌碼重心 POC
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # 3. ATR 與 RSI
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        df['Price_Chg'] = df['Close'].diff()
        gain = (df['Price_Chg'].where(df['Price_Chg'] > 0, 0)).rolling(window=14).mean()
        loss = (-df['Price_Chg'].where(df['Price_Chg'] < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # 4. 數據提取
        latest_p = float(df['Close'].iloc[-1])
        ma50_v = float(df['MA50'].iloc[-1])
        ma200_v = float(df['MA200'].iloc[-1])
        atr_v = float(df['ATR'].iloc[-1])
        cur_rsi = float(df['RSI'].iloc[-1])
        vol_v = float(df['Volume'].iloc[-1])
        vol_ma = float(df['Volume'].tail(20).mean())
        
        resistance = float(df['High'].tail(60).max())
        support = float(df['Low'].tail(60).min())
        
        pct_chg = ((latest_p / float(df['Close'].iloc[-2])) - 1) * 100
        vol_ratio = vol_v / vol_ma
        
        # 盈虧比分析
        risk_dist = latest_p - (latest_p - 2 * atr_v)
        reward_dist = resistance - latest_p
        rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 0

        # --- 組裝 AI 摘要 ---
        data_summary = (
            f"- 價格狀態: 現價 {latest_p:.2f}, MA50 {ma50_v:.2f}, MA200 {ma200_v:.2f}\n"
            f"- 漲跌/成交: 今日 {pct_chg:+.2f}%, 成交量倍數 {vol_ratio:.2f}x\n"
            f"- 籌碼: POC {poc_price:.2f}, 60日區間 [{support:.2f} - {resistance:.2f}]\n"
            f"- 指標: RSI {cur_rsi:.1f}, ATR {atr_v:.2f}\n"
            f"- 策略數據: 距離壓力位 {((resistance/latest_p)-1)*100:+.1f}%, 盈虧比 {rr_ratio:.2f}"
        )

        ai_note = get_ai_comment(ticker, data_summary)

        # --- 格式化報告 ---
        curr = "HK$ " if ".HK" in ticker else "$ "
        trend_tag = "📈 多頭排列" if latest_p > ma50_v > ma200_v else "📉 空頭結構" if latest_p < ma50_v < ma200_v else "🔄 區間震盪"
        
        report = (
            f"📊 **{ticker} 終極診斷報告**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 **現價**：{curr}{latest_p:.2f} ({pct_chg:+.2f}%)\n"
            f"🏗️ **形態**：{trend_tag}\n"
            f"📍 **籌碼重心**：{curr}{poc_price:.2f}\n"
            f"🛡️ **ATR 止損**：{curr}{latest_p - 2*atr_v:.2f}\n"
            f"⚖️ **盈虧比**：{rr_ratio:.2f}\n\n"
            f"🧠 **AI 深度策略推演**：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # --- 增強版繪圖 ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={'height_ratios': [2.5, 1]})
        plot_df = df.tail(120)
        
        # 主圖：價格與均線
        ax1.plot(plot_df.index, plot_df['Close'], color='black', lw=1.5, label='Price')
        ax1.plot(plot_df.index, plot_df['MA50'], color='blue', lw=1, alpha=0.8, label='MA50')
        ax1.plot(plot_df.index, plot_df['MA200'], color='purple', lw=1, alpha=0.8, label='MA200')
        ax1.axhline(poc_price, color='orange', ls='--', alpha=0.6, label='POC')
        ax1.axhline(resistance, color='red', ls=':', alpha=0.4, label='Resistance')
        ax1.fill_between(plot_df.index, plot_df['MA50'], plot_df['MA200'], color='lavender', alpha=0.2)
        ax1.set_title(f"{ticker} Technical Framework")
        ax1.legend(loc='upper left', fontsize='8')

        # 副圖：成交量與 RSI (雙軸可視化)
        ax2.bar(plot_df.index, plot_df['Volume'], color='gray', alpha=0.3, label='Volume')
        ax2_rsi = ax2.twinx()
        ax2_rsi.plot(plot_df.index, plot_df['RSI'], color='brown', lw=1, label='RSI')
        ax2_rsi.axhline(70, color='r', ls='--', alpha=0.3)
        ax2_rsi.axhline(30, color='g', ls='--', alpha=0.3)
        ax2.set_ylabel('Volume')
        ax2_rsi.set_ylabel('RSI')
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
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
        temp_msg = await update.message.reply_text(f"🚀 啟動終極分析引擎 {ticker}...")
        report, chart = await analyze_stock(ticker)
        if chart:
            await update.message.reply_photo(photo=chart, caption=report, parse_mode='Markdown')
        else:
            await update.message.reply_text(report)
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💹 歡迎使用終極 AI 診斷系統。\n直接傳送代號，我將為您解析籌碼、均線與風險報酬比。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 找不到 TELEGRAM_TOKEN")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 終極 AI 機器人已上線 (MA + ATR + Vol)...")
        app.run_polling(drop_pending_updates=True)
