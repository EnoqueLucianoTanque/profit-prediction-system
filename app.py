import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIGURAÇÃO
# =========================
st.set_page_config(
    page_title="Previsão de Lucro",
    page_icon="📈",
    layout="wide"
)

# =========================
# CSS (APENAS VISUAL)
# =========================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.main-title {
    font-size:42px;
    font-weight:bold;
    color:#ffffff;
}

.subtitle {
    font-size:16px;
    color:#b0b0b0;
}

.block-container {
    padding-top: 2rem;
}

/* cards */
.card {
    background-color:#1c1f26;
    padding:20px;
    border-radius:12px;
    color:white;
    box-shadow:0px 2px 10px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODELO
# =========================
modelo = joblib.load("modelo_xgb.pkl")

# =========================
# HEADER (MESMA LÓGICA)
# =========================
st.markdown('<p class="main-title">📊 Sistema Inteligente de Previsão de Lucro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Dashboard Analítico de Machine Learning</p>', unsafe_allow_html=True)

st.divider()

# =========================
# LAYOUT 2 COLUNAS (BI STYLE)
# =========================
col_inputs, col_preview = st.columns([1, 2])

# =========================
# INPUTS (SIDEBAR VISUAL MAIN)
# =========================
with col_inputs:

    st.markdown("## ⚙️ Parâmetros")

    marketing = st.number_input("Marketing", value=30000.0)
    preco = st.number_input("Preço Unitário", value=1200.0)
    clientes = st.number_input("Clientes", value=200)
    quantidade = st.number_input("Quantidade Vendida", value=50)
    stock = st.number_input("Nível de Stock", value=500)
    desconto = st.number_input("Desconto (%)", value=10.0)

# =========================
# PREVIEW + KPIs
# =========================
with col_preview:

    st.markdown("## 📋 Visão Geral")

    df_preview = pd.DataFrame([{
        "Marketing": marketing,
        "Preço": preco,
        "Clientes": clientes,
        "Qtd Vendida": quantidade,
        "Stock": stock,
        "Desconto": desconto
    }])

    st.dataframe(df_preview, use_container_width=True)

    st.divider()

    # =========================
    # PREVISÃO (MESMA LÓGICA)
    # =========================
    if st.button("🚀 Calcular Previsão", use_container_width=True):

        novos_dados = pd.DataFrame([{
            "Marketing": marketing,
            "Preco_Unitario": preco,
            "Clientes": clientes,
            "Quantidade_Vendida": quantidade,
            "Nivel_Stock": stock,
            "Desconto": desconto
        }])

        previsao = modelo.predict(novos_dados)[0]

        # =========================
        # KPI STYLE (SÓ VISUAL)
        # =========================
        col1, col2 = st.columns(2)

        col1.markdown(
            f"""
            <div class="card">
                <h3>💰 Lucro Previsto</h3>
                <h2>MZN {previsao:,.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col2.markdown(
            f"""
            <div class="card">
                <h3>📊 Status do Modelo</h3>
                <h2>XGBoost</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(f"💰 O lucro estimado é de MZN {previsao:,.2f}")

st.divider()

# =========================
# RODAPÉ
# =========================
st.caption("Projeto de Machine Learning desenvolvido por Enoque Luciano Tanque")