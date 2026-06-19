import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ====================================
# CONFIGURAÇÃO
# ====================================

st.set_page_config(
    page_title="Intelligent Profit Forecasting System",
    page_icon="📈",
    layout="wide"
)

# ====================================
# CSS PREMIUM (COM CARDS COLORIDOS)
# ====================================

st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top, #0B1220, #050A14);
    color: white;
}

/* TITULO */
.main-title {
    font-size: 46px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 15px;
    margin-bottom: 25px;
}

/* BASE CARD */
.card {
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.06);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-4px);
}

/* CORES DOS CARDS */
.card-blue {
    background: linear-gradient(145deg, #1E3A8A, #0B1220);
}

.card-green {
    background: linear-gradient(145deg, #065F46, #0B1220);
}

.card-purple {
    background: linear-gradient(145deg, #5B21B6, #0B1220);
}

.card-orange {
    background: linear-gradient(145deg, #9A3412, #0B1220);
}

/* BOTÃO */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #3B82F6, #2563EB);
    color: white;
    font-weight: bold;
    padding: 12px;
    border-radius: 12px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 6px 20px rgba(59,130,246,0.4);
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    background-color: #0B1220;
    border-radius: 12px;
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

col_left, col_right = st.columns([1, 2])

with col_left:

    st.subheader("⚙️ Business Parameters")
    st.markdown("---")

    marketing = st.number_input("Marketing", value=30000.0)
    preco = st.number_input("Preço Unitário", value=1200.0)
    clientes = st.number_input("Clientes", value=200)
    quantidade = st.number_input("Quantidade Vendida", value=50)
    stock = st.number_input("Nível de Stock", value=500)
    desconto = st.number_input("Desconto (%)", value=10.0)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🚀 CALCULAR PREVISÃO")

# ====================================
# PREVIEW
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

    st.dataframe(preview, use_container_width=True)

# ====================================
# KPIs MODERNOS (CORES DIFERENTES)
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

st.divider()

# ====================================
# PREVISÃO
# ====================================

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

    st.subheader("💰 Resultado da Previsão")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Lucro Previsto", f"MZN {previsao:,.2f}")

    with c2:
        st.metric("Modelo", "XGBoost")

    st.success(f"Lucro estimado: MZN {previsao:,.2f}")

    # ====================================
    # PERFORMANCE
    # ====================================

    st.divider()
    st.subheader("🤖 Performance do Modelo")

    m1, m2 = st.columns(2)

    with m1:
        st.metric("R² Score", "0.81")

    with m2:
        st.metric("MAE", "5,960")

    # ====================================
    # FEATURE IMPORTANCE
    # ====================================

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
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # ====================================
    # INTERPRETAÇÃO
    # ====================================

    top_feature = importancias.sort_values(
        "Importância", ascending=False
    ).iloc[0]["Variável"]

    st.info(f"🔍 Fator mais influente: {top_feature}")

    # ====================================
    # SOBRE O MODELO
    # ====================================

    st.divider()
    st.subheader("📚 Sobre o Modelo")

    st.markdown("""
    ### Modelo Utilizado
    XGBoost Regressor

    ### Objetivo
    Prever lucro empresarial com base em variáveis operacionais.

    ### Tecnologias
    Python • Pandas • Scikit-Learn • XGBoost • Streamlit • Plotly

    ### Aplicação
    Simulação de cenários de negócio e apoio à decisão estratégica.
    """)

# ====================================
# FOOTER
# ====================================

st.divider()

st.caption("Developed by Enoque Luciano Tanque | ML • XGBoost • Streamlit")