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

# --- LISTA DE ACTIVOS ---
assets = {
    "Índices (24/5)": {"Nasdaq 100": "NQ=F", "S&P 500": "ES=F", "DAX 40": "FDAX.EX"},
    "Currencies (24/5)": {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "JPY=X"},
    "Commodities (24/5)": {"Oro": "GC=F", "Petróleo": "CL=F"},
    "Crypto (24/7)": {"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"}
}

st.sidebar.title("📑 Master Sniper v11.7")
cat = st.sidebar.selectbox("Categoría", list(assets.keys()))
nombre = st.sidebar.selectbox("Activo", list(assets[cat].keys()))
temp = st.sidebar.selectbox("Temporalidad", ["1h", "4h", "1d"])
data = get_final_data(assets[cat][nombre], temp)

if data is not None:
    row = data.iloc[-1]
    # Pestañas originales solicitadas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Sniper Ejecución", "🕵️ Diagnóstico", "🧬 Historial Flujo", "🔗 Absorción Pro", "🏰 Camarilla Marks", "🧮 RISK MGR"
    ])

    with tab1:
        st.subheader(f"Centro de Operaciones - {nombre}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Z-Diff", f"{row['Z_Diff']:.2f}")
        c2.metric("Skewness", f"{row['Skew']:.2f}")
        c3.metric("R2 Calidad", f"{row['R2']:.3f}")
        
        # Lógica de escaneo temporal para la tarjeta principal
        last_sig = None
        for i in range(len(data)-1, len(data)-50, -1):
            if abs(data['Z_Diff'].iloc[i]) > 1.0 and data['R2'].iloc[i] > 0.05:
                delta = datetime.now(data.index[i].tzinfo) - data.index[i]
                last_sig = {
                    "tipo": "LONG (COMPRA)" if data['Z_Diff'].iloc[i] < -1.0 else "SHORT (VENTA)",
                    "precio": data['Close'].iloc[i],
                    "hace": int(delta.total_seconds() / 3600),
                    "hora": data.index[i].strftime("%H:%M"),
                    "z": data['Z_Diff'].iloc[i], "r2": data['R2'].iloc[i]
                }
                break

        if last_sig:
            color = "#00ff00" if "LONG" in last_sig['tipo'] else "#ff4b4b"
            txt_tiempo = f"hace {last_sig['hace']} horas" if last_sig['hace'] > 0 else "¡AHORA MISMO!"
            prob = min(50.0 + abs(last_sig['z'])*12 + last_sig['r2']*45, 98.4)
            st.markdown(f"""<div class="signal-card" style="border-color: {color};"><h2 style="color: {color};">🔥 ÚLTIMA SEÑAL: {last_sig['tipo']}</h2><div style="display: flex; justify-content: space-between;"><div><p>Activada: <b>{txt_tiempo}</b> ({last_sig['hora']})</p><p>Precio Entrada: <b>{last_sig['precio']:.4f}</b></p></div><div style="text-align: right;"><p>Probabilidad</p><h1 style="color: {color};">{prob:.1f}%</h1></div></div></div>""", unsafe_allow_html=True)
        else:
            st.info("📉 Esperando confluencia institucional...")
        st.plotly_chart(go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])]).update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab2:
        st.subheader("🕵️ Centro de Diagnóstico Dinámico")
        st.table(get_dynamic_diagnosis(row['Z_Diff'], row['Z_Price'], row['Skew'], row['R2']))

    with tab3:
        st.markdown("<div class='gold-header'>🧬 HISTORIAL DE FLUJO INSTITUCIONAL</div>", unsafe_allow_html=True)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Price'], name="Precio (Z)", line=dict(color='#00d4ff')))
        fig_f.add_trace(go.Scatter(x=data.index, y=data['Z_Diff'], name="Flujo (Z)", line=dict(color='#ffd700', dash='dot')))
        st.plotly_chart(fig_f.update_layout(template="plotly_dark", height=450), use_container_width=True)

    with tab4:
        st.markdown("<div class='gold-header'>🔗 MASTER DE ABSORCIÓN INSTITUCIONAL</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 2])
        with col_a:
            if row['Z_Eff'] > 1.5: st.success("*ALTA EFICIENCIA:* El precio fluye.")
            elif row['Z_Eff'] < -1.5: st.warning("*ABSORCIÓN:* Volumen frenando el precio.")
            else: st.write("Flujo normal.")
        with col_b:
            st.plotly_chart(px.bar(data.tail(40), y='Z_Eff', color='Z_Eff', color_continuous_scale='RdYlGn').update_layout(template="plotly_dark", height=350), use_container_width=True)

    with tab5:
        st.markdown("<div class='gold-header'>🏰 NIVELES CAMARILLA CON MARCAS DE SEÑAL</div>", unsafe_allow_html=True)
        df_p = data.tail(100)
        fig_c = go.Figure(data=[go.Candlestick(x=df_p.index, open=df_p['Open'], high=df_p['High'], low=df_p['Low'], close=df_p['Close'], name="Precio")])
        for n, c in [('H4', 'red'), ('H3', 'orange'), ('L3', 'lightgreen'), ('L4', 'green')]:
            fig_c.add_hline(y=row[n], line_dash="dash", line_color=c, annotation_text=n)
        for i in range(len(df_p)):
            if abs(df_p['Z_Diff'].iloc[i]) > 1.0 and df_p['R2'].iloc[i] > 0.05:
                if df_p['Z_Diff'].iloc[i] < -1.0: # BUY
                    fig_c.add_annotation(x=df_p.index[i], y=df_p['Low'].iloc[i], text="▲", showarrow=False, font=dict(color="#00ff00", size=18), yshift=-20)
                else: # SELL
                    fig_c.add_annotation(x=df_p.index[i], y=df_p['High'].iloc[i], text="▼", showarrow=False, font=dict(color="#ff4b4b", size=18), yshift=20)
        st.plotly_chart(fig_c.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False), use_container_width=True)

    with tab6:
        st.subheader("🧮 Risk Manager (RoboForex)")
        balance = st.number_input("Balance (USD)", value=1000.0)
        riesgo_pct = st.slider("Riesgo %", 0.1, 5.0, 1.0)
        sl_pips = st.number_input("Stop Loss Pips", value=10.0)
        riesgo_usd = balance * (riesgo_pct / 100)
        st.metric("Pérdida Máxima", f"${riesgo_usd:.2f}")
        st.metric("Lotaje Sugerido", f"{round(riesgo_usd / (sl_pips * 10), 2)}")

else:
    st.error("Error al cargar datos financieros.")
