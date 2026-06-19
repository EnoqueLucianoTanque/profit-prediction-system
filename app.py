import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# ====================================
# CONFIGURAÇÃO
# ====================================

st.set_page_config(
    page_title="Intelligent Profit Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ====================================
# CSS
# ====================================

st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top, #0B1220, #050A14);
    color: white;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.4rem;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 14px;
    margin-bottom: 15px;
}

.card {
    padding: 12px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.06);
}

.card-blue { background: linear-gradient(145deg, #1E3A8A, #0B1220); }
.card-green { background: linear-gradient(145deg, #065F46, #0B1220); }
.card-purple { background: linear-gradient(145deg, #5B21B6, #0B1220); }
.card-orange { background: linear-gradient(145deg, #9A3412, #0B1220); }

div[data-baseweb="input"] {
    margin-bottom: -10px !important;
}

[data-testid="stDataFrame"] {
    background-color: #0B1220;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ====================================
# MODELO
# ====================================

modelo = joblib.load("modelo_xgb.pkl")

# ====================================
# HEADER
# ====================================

st.markdown('<p class="main-title">📈 Intelligent Profit Forecasting System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning • XGBoost • Predictive Analytics • Business Intelligence</p>', unsafe_allow_html=True)

st.divider()

# ====================================
# INPUTS
# ====================================

col_left, col_right = st.columns([1, 1], vertical_alignment="top")

with col_left:

    st.subheader("⚙️ Business Parameters")

    c1, c2 = st.columns(2)

    with c1:
        marketing = st.number_input("Marketing", value=30000.0)
        clientes = st.number_input("Clientes", value=200)
        stock = st.number_input("Stock", value=500)

    with c2:
        preco = st.number_input("Preço Unitário", value=1200.0)
        quantidade = st.number_input("Quantidade Vendida", value=50)
        desconto = st.number_input("Desconto (%)", value=10.0)

    predict_btn = st.button("🚀 CALCULAR PREVISÃO", use_container_width=True)

# ====================================
# PREVIEW + RESULTADO
# ====================================

with col_right:

    st.subheader("📋 Dados de Entrada")

    preview = pd.DataFrame([{
        "Marketing": marketing,
        "Preço Unitário": preco,
        "Clientes": clientes,
        "Quantidade Vendida": quantidade,
        "Stock": stock,
        "Desconto": desconto
    }])

    st.dataframe(preview, use_container_width=True, height=140)

    if predict_btn:

        novos_dados = pd.DataFrame([{
            "Marketing": marketing,
            "Preco_Unitario": preco,
            "Clientes": clientes,
            "Quantidade_Vendida": quantidade,
            "Nivel_Stock": stock,
            "Desconto": desconto
        }])

        previsao = modelo.predict(novos_dados)[0]

        st.markdown("### 💰 Resultado da Previsão")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Lucro Previsto", f"MZN {previsao:,.2f}")

        with c2:
            st.metric("Modelo", "XGBoost")

        st.success(f"Lucro estimado: MZN {previsao:,.2f}")

# ====================================
# KPIs
# ====================================

st.divider()

k1, k2, k3, k4 = st.columns(4)

k1.markdown(f"""
<div class="card card-blue">
<h3>👥 Clientes</h3>
<h2>{clientes}</h2>
</div>
""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="card card-green">
<h3>📦 Quantidade</h3>
<h2>{quantidade}</h2>
</div>
""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="card card-purple">
<h3>💵 Preço</h3>
<h2>MZN {preco:,.0f}</h2>
</div>
""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="card card-orange">
<h3>📢 Marketing</h3>
<h2>MZN {marketing:,.0f}</h2>
</div>
""", unsafe_allow_html=True)

# ====================================
# PERFORMANCE + FEATURE IMPORTANCE + SIMULAÇÃO
# ====================================

if predict_btn:

    st.divider()
    st.subheader("🤖 Performance do Modelo")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("R² Score", "0.81")

    with c2:
        st.metric("MAE", "5,960")

    st.divider()
    st.subheader("📈 Impacto das Variáveis")

    features = [
        "Marketing",
        "Preço Unitário",
        "Clientes",
        "Quantidade Vendida",
        "Nível Stock",
        "Desconto"
    ]

    importancias = pd.DataFrame({
        "Variável": features,
        "Importância": modelo.feature_importances_
    }).sort_values("Importância", ascending=True)

    fig = px.bar(
        importancias,
        x="Importância",
        y="Variável",
        orientation="h",
        template="plotly_dark",
        height=420
    )

    st.plotly_chart(fig, use_container_width=True)

    top_feature = importancias.sort_values(
        "Importância", ascending=False
    ).iloc[0]["Variável"]

    st.info(f"🔍 Fator mais influente: {top_feature}")

    # ====================================
    # 📊 SIMULAÇÃO: ISOLADO vs COMBINADO
    # ====================================

    st.divider()
    st.subheader("📊 Simulação de Cenários de Lucro")

    col_a, col_b = st.columns(2)

    # 🔵 ISOLADO
    with col_a:

        st.markdown("### 🔵 Análise Isolada (Marketing)")

        marketing_range = np.linspace(marketing * 0.7, marketing * 1.5, 6)

        isolado = []

        for m in marketing_range:
            df = pd.DataFrame([{
                "Marketing": m,
                "Preco_Unitario": preco,
                "Clientes": clientes,
                "Quantidade_Vendida": quantidade,
                "Nivel_Stock": stock,
                "Desconto": desconto
            }])

            lucro = modelo.predict(df)[0]

            isolado.append({"Marketing": m, "Lucro": lucro})

        df_iso = pd.DataFrame(isolado)

        fig_iso = px.line(
            df_iso,
            x="Marketing",
            y="Lucro",
            markers=True,
            template="plotly_dark",
            title="Impacto isolado do Marketing"
        )

        st.plotly_chart(fig_iso, use_container_width=True)

    # 🟢 COMBINADO
    with col_b:

        st.markdown("### 🟢 Análise Combinada")

        combinado = []

        marketing_range = np.linspace(marketing * 0.7, marketing * 1.5, 6)
        clientes_range = np.linspace(clientes * 0.7, clientes * 1.5, 6)
        desconto_range = np.linspace(desconto * 0.5, desconto * 1.5, 6)

        for i in range(6):

            df = pd.DataFrame([{
                "Marketing": marketing_range[i],
                "Preco_Unitario": preco,
                "Clientes": clientes_range[i],
                "Quantidade_Vendida": quantidade,
                "Nivel_Stock": stock,
                "Desconto": desconto_range[i]
            }])

            lucro = modelo.predict(df)[0]

            combinado.append({
                "Cenario": f"Cenário {i+1}",
                "Lucro": lucro
            })

        df_com = pd.DataFrame(combinado)

        fig_com = px.line(
            df_com,
            x="Cenario",
            y="Lucro",
            markers=True,
            template="plotly_dark",
            title="Impacto combinado (Marketing + Clientes + Desconto)"
        )

        st.plotly_chart(fig_com, use_container_width=True)

# ====================================
# FOOTER
# ====================================

st.divider()

st.caption("Developed by Enoque Luciano Tanque | ML • XGBoost • Streamlit")