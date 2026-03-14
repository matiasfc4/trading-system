import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from datetime import datetime

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Alpha Quant v11.7 - Camarilla Marks", layout="wide")

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
    
    df['Vol_Proxy'] = (df['High'] - df['Low']) * 100000
    df['RMF'] = df['Close'] * df['Vol_Proxy']
    diff_val = df['Ret'].rolling(W).sum() - df['RMF'].pct_change().rolling(W).sum()
    df['Z_Diff'] = (diff_val - diff_val.rolling(W).mean()) / (diff_val.rolling(W).std() + 1e-10)
    
    r2_s = []
    for i in range(len(df)):
        if i < W: r2_s.append(0); continue
        sub = df.iloc[i-W:i].dropna()
        try: r2_s.append(sm.OLS(sub['Ret'], sm.add_constant(sub['RMF'])).fit().rsquared)
        except: r2_s.append(0)
    df['R2'] = r2_s

    # Niveles Camarilla (del día anterior)
    daily = yf.download(ticker_id, period="5d", interval="1d", progress=False)
    if isinstance(daily.columns, pd.MultiIndex): daily.columns = daily.columns.get_level_values(0)
    H, L, C = daily['High'].iloc[-2], daily['Low'].iloc[-2], daily['Close'].iloc[-2]
    r_val = H - L
    df['H4'], df['H3'] = C + r_val * (1.1/2), C + r_val * (1.1/4)
    df['L3'], df['L4'] = C - r_val * (1.1/4), C - r_val * (1.1/2)
    return df

# --- LISTA DE ACTIVOS ---
assets = {
    "Índices": {"Nasdaq 100": "NQ=F", "S&P 500": "ES=F"},
    "Currencies": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"},
    "Crypto": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"}
}

st.sidebar.title("📑 Master Sniper v11.7")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    tab1, tab5 = st.tabs(["🎯 Sniper Ejecución", "🏰 Camarilla Marks"])

    with tab1:
        # (Aquí iría tu lógica de la señal actual que ya tienes)
        st.info("Consulta la pestaña Camarilla para ver las marcas históricas.")
        st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])]).update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab5:
        st.subheader(f"Niveles y Señales Históricas - {nombre}")
        
        # 1. Crear el gráfico de velas (últimas 100 para que se vea bien)
        df_plot = data.tail(100)
        fig_cam = go.Figure(data=[go.Candlestick(
            x=df_plot.index, 
            open=df_plot['Open'], 
            high=df_plot['High'], 
            low=df_plot['Low'], 
            close=df_plot['Close'],
            name="Precio"
        )])

        # 2. Añadir líneas Camarilla
        for n, c in [('H4', 'red'), ('H3', 'orange'), ('L3', 'lightgreen'), ('L4', 'green')]:
            fig_cam.add_hline(y=row[n], line_dash="dash", line_color=c, annotation_text=n)

        # 3. ESCANEO Y MARCADO DE SEÑALES
        for i in range(len(df_plot)):
            z_val = df_plot['Z_Diff'].iloc[i]
            r2_val = df_plot['R2'].iloc[i]
            
            # Si hay confluencia, ponemos una marca
            if abs(z_val) > 1.0 and r2_val > 0.05:
                # LONG: Triángulo verde hacia arriba debajo del Low
                if z_val < -1.0:
                    fig_cam.add_annotation(
                        x=df_plot.index[i], y=df_plot['Low'].iloc[i],
                        text="▲", showarrow=False, font=dict(color="#00ff00", size=18),
                        yshift=-20
                    )
                # SHORT: Triángulo rojo hacia abajo encima del High
                else:
                    fig_cam.add_annotation(
                        x=df_plot.index[i], y=df_plot['High'].iloc[i],
                        text="▼", showarrow=False, font=dict(color="#ff4b4b", size=18),
                        yshift=20
                    )

        fig_cam.update_layout(
            height=600, 
            template="plotly_dark", 
            xaxis_rangeslider_visible=False,
            title="Marcas: ▲/▼ indican activación de señal institucional"
        )
        st.plotly_chart(fig_cam, use_container_width=True)

else:
    st.error("Error al conectar con la API.")
