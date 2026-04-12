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

# 分析參數
BINS_COUNT = 80
VOL_THRESHOLD = 2.0

# 設定日誌
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def get_ai_comment(ticker, price, rsi, poc, action, advice, rr_ratio):
    """呼叫 DeepSeek API 生成點評"""
    if not DEEPSEEK_KEY:
        return "⚠️ AI 分析未啟用 (請檢查 DEEPSEEK_API_KEY 設定)"
    
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_KEY.strip()}"
    }
    
    prompt = f"""
    作為專業資深交易員，請對股票 {ticker} 進行診斷：
    【關鍵數據】
    - 現價：{price}
    - 籌碼重心 (POC)：{poc}
    - RSI：{rsi}
    - 主力行為：{action}
    - 系統計算盈虧比：{rr_ratio:.2f}

    【分析任務】
    請依照以下格式，提供不超過 150 字的繁體中文分析：
    1. **趨勢定調**：判斷目前趨勢。
    2. **具體買入建議**：給出區間（參考 POC）。
    3. **目標與止損**：給出具體點位。
    4. **盈虧比評價**：針對 {rr_ratio:.2f} 的盈虧比給出操作評價。

    直接給結論，口吻要果斷專業。
    """
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一位精通籌碼分佈與技術分析的專業交易員。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"AI 暫時無法連線: {str(e)[:50]}"

async def analyze_stock(ticker):
    """核心分析函數"""
    try:
        df_raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        
        if df_raw.empty or len(df_raw) < 35:
            return f"❌ 無法獲取 {ticker} 的數據，請檢查代碼是否正確。", None

        df = df_raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 1. 技術指標計算
        prices, vols = df['Close'].values, df['Volume'].values
        hist, bin_edges = np.histogram(prices, BINS_COUNT, weights=vols)
        poc_price = ((bin_edges[:-1] + bin_edges[1:]) / 2)[np.argmax(hist)]

        df['Price_Chg'] = df['Close'].diff()
        df['MF'] = np.where(df['Price_Chg'] > 0, df['Volume'], np.where(df['Price_Chg'] < 0, -df['Volume'], 0))
        df['Cum_MF'] = df['MF'].cumsum()

        delta = df['Price_Chg']
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)

        # 2. 提取數值
        latest_p = float(df['Close'].iloc[-1])
        pct_chg = ((latest_p / float(df['Close'].iloc[-2])) - 1) * 100
        vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].tail(20).mean())
        cur_rsi = float(df['RSI'].iloc[-1])
        
        # 3. 盈虧比 (Risk/Reward) 計算
        # 假設目標位為近期 60 天高點，止損位為 POC 下方 3%
        recent
