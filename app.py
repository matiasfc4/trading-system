import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v11.7 - Full Restoration", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #0d1117; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .diag-box { background-color: #161b22; border-left: 5px solid #ffd700; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .gold-header { color: #ffd700; font-weight: bold; border-bottom: 1px solid #ffd700; padding-bottom: 5px; margin-bottom: 15px; }
    .camarilla-box { background-color: #0a0e14; border: 1px solid #444; padding: 10px; border-radius: 5px; text-align: center; }
    .signal-card { background: linear-gradient(135deg, #1e2530 0%, #0d1117 100%); border: 2px solid #30363d; padding: 25px; border-radius: 15px; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)
def get_final_data(ticker_id, t):
    p_map = {"1h": "30d", "4h": "60d", "1d": "120d"}
    df = yf.download(ticker_id, period=p_map[t], interval=t, progress=False)
    
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    
    W = 14
    df['Ret'] = df['Close'].pct_change()
    df['SMA'] = df['Close'].rolling(W).mean()
    df['Std'] = df['Close'].rolling(W).std()
    df['Z_Price'] = (df['Close'] - df['SMA']) / (df['Std'] + 1e-10)
    
    # JDetector (Flujo Institucional)
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(W).sum() - df['RMF'].pct_change().rolling(W).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(W).mean()) / (diff_val.rolling(W).std() + 1e-10)
    
    # Estructura y Absorción
    df['Skew'] = df['Ret'].rolling(30).skew()
    df['Spread'] = (df['High'] - df['Low'])
    df['V_Eff'] = df['Spread'] / (df['Volume'].rolling(5).mean() + 1e-10)
    df['Z_Eff'] = (df['V_Eff'] - df['V_Eff'].rolling(W).mean()) / (df['V_Eff'].rolling(W).std() + 1e-10)
    
    r2_s = []
    for i in range(len(df)):
        if i < W: r2_s.append(0); continue
        sub = df.iloc[i-W:i].dropna()
        try: r2_s.append(sm.OLS(sub['Ret'], sm.add_constant(sub['RMF'])).fit().rsquared)
        except: r2_s.append(0)
    df['R2'] = r2_s

    # Camarilla
    daily = yf.download(ticker_id, period="5d", interval="1d", progress=False)
    if isinstance(daily.columns, pd.MultiIndex): daily.columns = daily.columns.get_level_values(0)
    H, L, C = daily['High'].iloc[-2], daily['Low'].iloc[-2], daily['Close'].iloc[-2]
    r_val = H - L
    df['H4'], df['H3'] = C + r_val * (1.1/2), C + r_val * (1.1/4)
    df['L3'], df['L4'] = C - r_val * (1.1/4), C - r_val * (1.1/2)
    
    return df

def get_dynamic_diagnosis(z_d, z_p, skew, r2):
    diag = []
    if z_d < -1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🟢 COMPRA", "Significado": "Entrada de dinero institucional"})
    elif z_d > 1.0: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "🔴 VENTA", "Significado": "Salida de dinero / Distribución"})
    else: diag.append({"Dato": "Z-Diff (Flujo)", "Estado": "⚪ Neutral", "Significado": "Sin presión clara"})
    
    if abs(z_p) > 2: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚠️ EXTREMO", "Significado": "Precio sobreextendido. Reversión probable."})
    else: diag.append({"Dato": "Z-Price (Nivel)", "Estado": "⚓ Estable", "Significado": "Zona de Fair Value"})
    
    if skew > 0.2: diag.append({"Dato": "Skewness", "Estado": "🚀 Alcista", "Significado": "Sesgo de rebote rápido"})
    elif skew < -0.2: diag.append({"Dato": "Skewness", "Estado": "📉 Bajista", "Significado": "Riesgo de caídas bruscas"})
    else: diag.append({"Dato": "Skewness", "Estado": "⚖️ Simétrico", "Significado": "Equilibrio de riesgo"})
    
    if r2 > 0.15: diag.append({"Dato": "R2 (Calidad)", "Estado": "💎 ALTA", "Significado": "Movimiento institucional confirmado"})
    else: diag.append({"Dato": "R2 (Calidad)", "Estado": "💨 RUIDO", "Significado": "Cuidado con trampas de bajo volumen"})
    return pd.DataFrame(diag)

assets = {
    "Índices (24/5 Continuos)": {"Nasdaq 100 E-Mini": "NQ=F", "S&P 500 E-Mini": "ES=F", "Dow Jones Mini": "YM=F", "DAX 40 (GER)": "FDAX.EX", "Nikkei 225": "NKD=F"},
    "Currencies (24/5)": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X", "AUD/USD": "AUDUSD=X"},
    "Commodities (24/5)": {"Oro": "GC=F", "Plata": "SI=F", "Petróleo WTI": "CL=F", "Cobre": "HG=F"},
    "Crypto (24/7)": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"}
}

st.sidebar.title("📑 Master Sniper v11.7")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Sniper Ejecución", "🕵️ Diagnóstico", "🧬 Historial Flujo", "🔗 Absorción Pro", "🏰 Camarilla Marks", "🧮 RISK MGR"
    ])

    with tab1:
        st.subheader(f"Centro de Operaciones - {nombre}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Diff", f"{row['Z_Diff']:.2f}")
        c2.metric("Skewness", f"{row['Skew']:.2f}")
        c3.metric("R2 Calidad", f"{row['R2']:.3f}")
        
        last_signal = None
        for i in range(len(data)-1, len(data)-50, -1):
            if i < 0: break
            if abs(data['Z_Diff'].iloc[i]) > 1.0 and data['R2'].iloc[i] > 0.05:
                # Normalizar zona horaria para el cálculo
                dt_now = datetime.now(data.index[i].tzinfo)
                delta = dt_now - data.index[i]
                last_signal = {
                    "tipo": "LONG (COMPRA)" if data['Z_Diff'].iloc[i] < -1.0 else "SHORT (VENTA)",
                    "precio": data['Close'].iloc[i],
                    "
