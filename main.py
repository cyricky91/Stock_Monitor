import logging
import re
import io
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# 1. 基礎設定
# 優先從 Railway 的 Variables 讀取 Token，如果沒有則使用預設值
TOKEN = os.getenv("TELEGRAM_TOKEN")

# 分析參數設定
BINS_COUNT = 80
VOL_THRESHOLD = 2.0

# 設定日誌
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def analyze_stock(ticker):
    """
    核心分析函數：處理數據、計算指標、生成報告與圖表
    """
    try:
        # 下載數據 (使用單一 ticker 模式)
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 35:
            return f"❌ 無法獲取 {ticker} 的足夠數據，請檢查代碼是否正確。", None

        # --- 數據清洗：徹底解決 float() argument must be a string or a real number, not 'Series' 報錯 ---
        df = df_raw.copy()
        # 如果 yfinance 回傳了多重索引 (MultiIndex)，強制簡化為單層索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 再次檢查必要欄位，防止 MultiIndex 處理後名稱異常
        if 'Close' not in df.columns or 'Volume' not in df.columns:
             return f"❌ {ticker} 數據欄位異常，請稍後再試。", None

        # 2. 籌碼重心 (POC)
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, bins=bins_count, weights=vols)
        poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # 累計資金流 (Cumulative Money Flow)
        df['Price_Chg'] = df['Close'].diff()
        df['MF'] = np.where(df['Price_Chg'] > 0, df['Volume'], np.where(df['Price_Chg'] < 0, -df['Volume'], 0))
        df['Cum_MF'] = df['MF'].cumsum()

        # RSI & 布林通道
        delta = df['Price_Chg']
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
            
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)

        # 3. 提取數值 (確保提取的是純數值標量)
        latest_p = float(df['Close'].iloc[-1])
        prev_p = float(df['Close'].iloc[-2])
        pct_chg = ((latest_p / prev_p) - 1) * 100
        vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].tail(20).mean())
        cur_rsi = float(df['RSI'].iloc[-1])
        
        # 4. 判斷市場與策略邏輯
        is_hk = ".HK" in ticker
        curr = "HK$ " if is_hk else "$ "
        
        # 主力行為判定
        main_action = "🔥 主力放量進場" if vol_ratio >= VOL_THRESHOLD and pct_chg > 1.8 else \
                      "😱 主力放量派發" if vol_ratio >= VOL_THRESHOLD and pct_chg < -1.8 else "🔘 籌碼縮量整理"
        
        # 交易建議判定
        if latest_p > poc_price:
            advice = "📈 多頭結構，建議分批佈局" if cur_rsi < 68 else "⚠️ 漲幅過快，暫不追高"
        else:
            advice = "📉 弱勢壓制，建議觀望" if cur_rsi > 35 else "🔵 超跌區域，靜待反彈"

        # 5. 組裝深度診斷報告
        report = (
            f"📊 **{ticker} 深度分析報告**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 **現價**：{curr}{latest_p:.2f} ({pct_chg:+.2f}%)\n"
            f"📍 **籌碼重心**：{curr}{poc_price:.2f}\n"
            f"🔍 **主力行為**：{main_action}\n"
            f"💡 **交易建議**：{advice}\n\n"
            f"🎯 **操作參考**：\n"
            f"  - 買入支撐位：{curr}{poc_price:.2f}\n"
            f"  - 止損參考位：{curr}{poc_price * 0.95:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # 6. 視覺化繪圖 (優化手機顯示效果)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # 上圖：股價與 POC
        ax1.plot(df.index[-100:], df['Close'].tail(100), color='#1f77b4', lw=2, label='Price')
        ax1.axhline(poc_price, color='red', ls='--', alpha=0.7, label=f'POC: {poc_price:.2f}')
        ax1.set_title(f"{ticker} Trend Analysis")
        ax1.legend(loc='upper left')
        
        # 下圖：資金流
        ax2.fill_between(df.index[-100:], df['Cum_MF'].tail(100), color='purple', alpha=0.1)
        ax2.plot(df.index[-100:], df['Cum_MF'].tail(100), color='purple', label='Money Flow Strength')
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        
        # 轉存為二進制流以便發送
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        return f"❌ 分析出錯: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    處理使用者傳送的文字訊息
    """
    if not update.message or not update.message.text:
        return
        
    text = update.message.text.upper().strip()
    
    # 解析輸入：支援多個代碼、自動補齊港股
    tickers = [s.zfill(4)+".HK" if s.isdigit() else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp_msg = await update.message.reply_text(f"🔍 正在深度診斷 {ticker}，請稍候...")
        
        report, chart = await analyze_stock(ticker)
        
        if chart:
            await update.message.reply_photo(photo=chart, caption=report, parse_mode='Markdown')
        else:
            await update.message.reply_text(report)
            
        # 刪除「正在診斷」的提示，保持對話整潔
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    處理 /start 指令
    """
    await update.message.reply_text(
        "👋 你好！我是你的籌碼診斷助手。\n\n"
        "📍 **如何使用：**\n"
        "直接傳送股票代號給我即可，例如：\n"
        "• 港股：`0005` 或 `700`\n"
        "• 美股：`NVDA` 或 `TSLA`\n\n"
        "我會立刻為你分析 **主力成本區 (POC)** 與 **操作建議**！"
    )

if __name__ == '__main__':
    # 啟動機器人
    app = Application.builder().token(TOKEN).build()
    
    # 註冊處理器
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 即時診斷機器人已啟動，監聽中...")
    app.run_polling()
