import logging
import re
import io
impocurr}{resist requests
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
VOL_THRESHOLD = 2.0

# 設置日誌
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_ai_comment(ticker, data_summary):
    """呼叫 DeepSeek API，增加超時處理與詳細報錯"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用：請在 Railway 設置 DEEPSEEK_API_KEY。"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"""
    作為量化資產管理主管，請針對 {ticker} 進行診斷：
    【數據包】: {data_summary}
    【任務】: 提供 250-300 字繁體中文分析。包括：
    1. 籌碼結構與 POC 意義。
    2. 動能與 RSI/MACD 背離偵測。
    3. 基於 ATR 的進場與防守方案。
    4. 1-10分評分。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位專業的對沖基金策略師，說話精準、不廢話。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        # 設置 25 秒超時，防止 Railway 任務卡死
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.Timeout:
        return "⚠️ AI 分析超時：DeepSeek 伺服器回應過慢，請稍後再試。"
    except Exception as e:
        logger.error(f"AI API Error: {str(e)}")
        return f"⚠️ AI 診斷連線失敗：請檢查 API 餘額或 Key 設定。原因: {str(e)[:50]}"

async def analyze_stock(ticker):
    """核心分析函數，增加數據下載檢查"""
    try:
        # 下載數據，設置超時
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 60:
            return f"❌ 數據下載失敗：找不到股票代號 {ticker} 或數據不足。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # --- 技術指標計算 ---
        # 均線
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Hist'] = df['MACD'] - df['Signal']

        # ATR & RSI
        tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))

        # POC 籌碼重心
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_p = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        # --- 數據摘要 ---
        last = df.iloc[-1]
        latest_p = float(last['Close'])
        atr_v = float(last['ATR'])
        resis = float(df['High'].tail(60).max())
        
        data_summary = (
            f"現價:{latest_p:.2f}, POC:{poc_p:.2f}, RSI:{last['RSI']:.1f}, "
            f"MACD:{last['MACD']:.3f}, ATR:{atr_v:.2f}, MA50:{last['MA50']:.2f}"
        )

        ai_note = get_ai_comment(ticker, data_summary)

        # --- 報告組裝 ---
        curr = "HK$ " if ".HK" in ticker else "$ "
        report = (
            f"🚀 *{ticker} 診斷報告*\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💰 *現價*：{curr}{latest_p:.2f}\n"
            f"📍 *籌碼重心*：{curr}{poc_p:.2f}\n"
            f"🛡️ *ATR 止損位*：{curr}{latest_p - 2.5*atr_v:.2f}\n"
            f"🎯 *壓力位*：{curr}{resis:.2f}\n\n"
            f"🧠 *AI 策略分析*：\n{ai_note}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )

        # --- 繪圖 ---
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})
        plot_df = df.tail(100)
        ax1.plot(plot_df.index, plot_df['Close'], color='black', lw=1.5, label='Price')
        ax1.plot(plot_df.index, plot_df['MA50'], label='MA50', alpha=0.7)
        ax1.axhline(poc_p, color='orange', ls='--', label='POC')
        ax1.set_title(f"{ticker} Analysis")
        ax1.legend()

        ax2.bar(plot_df.index, plot_df['Hist'], color='gray', alpha=0.3, label='MACD Hist')
        ax2.plot(plot_df.index, plot_df['MACD'], label='MACD')
        ax2.plot(plot_df.index, plot_df['Signal'], label='Signal')
        ax2.legend()
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        plt.close(fig)
        
        return report, buf
    except Exception as e:
        logger.error(f"Analysis Error: {str(e)}")
        return f"❌ 數據處理出錯: {str(e)}", None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    text = update.message.text.upper().strip()
    # 自動補全港股代號
    tickers = [s.zfill(4)+".HK" if s.isdigit() and len(s)<=4 else s for s in re.split(r'[，,；;\s]+', text)]
    
    for ticker in tickers:
        temp_msg = await update.message.reply_text(f"🔋 正在分析 {ticker}...")
        report, chart = await analyze_stock(ticker)
        
        try:
            if chart:
                try:
                    # 優先嘗試以 Markdown 模式發送
                    await update.message.reply_photo(photo=chart, caption=report, parse_mode=ParseMode.MARKDOWN)
                except Exception as e:
                    logger.warning(f"Markdown parse failed, falling back to plain text: {e}")
                    # 如果解析失敗，則以普通文本發送，防止沒反應
                    await update.message.reply_photo(photo=chart, caption=report)
            else:
                await update.message.reply_text(report)
        except Exception as final_e:
            await update.message.reply_text(f"❌ 消息發送失敗: {str(final_e)}")
        
        await temp_msg.delete()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 系統已就緒。傳送股票代號（如 700 或 AAPL）獲取旗艦診斷。")

if __name__ == '__main__':
    if not TOKEN:
        print("❌ 錯誤: 未設置 TELEGRAM_TOKEN 環境變數")
    else:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        print("🤖 機器人正在運行中...")
        app.run_polling(drop_pending_updates=True)
