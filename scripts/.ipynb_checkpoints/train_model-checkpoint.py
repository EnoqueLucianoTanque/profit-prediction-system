import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib


# carregar excel
df = pd.read_excel(
    "data/raw/planilha_analise_financeira_ml.xlsx"
)

# mostrar dados
print(df.head())

# variáveis de entrada
X = df[
    [
        "Marketing",
        "Preco_Unitario",
        "Clientes",
        # "Inflacao",
        "Quantidade_Vendida",
        "Nivel_Stock",
        "Desconto",
        # "Quantidade_Vendida",
        # "Receita",
        # "Custo",
        # "Meta_Vendas"
    ]
]

# variável alvo
y = df["Lucro"]

# dividir treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

            # REgressao Linear

# criar modelo
modelo = LinearRegression()

# treinar
modelo.fit(X_treino, y_treino)

# previsões
previsoes = modelo.predict(X_teste)

# métricas
erro = mean_absolute_error(y_teste, previsoes)

r2 = r2_score(y_teste, previsoes)

print("\nRESULTADOS")

print("Erro Médio:", erro)

print("R2:", r2)



plt.scatter(y_teste, previsoes)

plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")

plt.title("Real vs Previsto")

plt.show()


# coeficientes
coeficientes = pd.DataFrame(
    modelo.coef_,
    X.columns,
    columns=["Coeficiente"]
)

print("\nIMPORTÂNCIA DAS VARIÁVEIS")

print(coeficientes)

print(df.isnull().sum())




            #Random Forest

modelo_rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

modelo_rf.fit(X_treino, y_treino)

pred_rf = modelo_rf.predict(X_teste)

print("\n🌳 RANDOM FOREST")
print("R2:", r2_score(y_teste, pred_rf))
print("MAE:", mean_absolute_error(y_teste, pred_rf))


print("\n🔥 COMPARAÇÃO FINAL")

print("Random Forest R2:", r2_score(y_teste, pred_rf))

print("Random Forest MAE:", mean_absolute_error(y_teste, pred_rf))



plt.scatter(y_teste, pred_rf)

plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos (Random Forest)")
plt.title("Real vs Previsto - Random Forest")

plt.show()


importances = pd.DataFrame({
    "Feature": X.columns,
    "Importância": modelo_rf.feature_importances_
}).sort_values(by="Importância", ascending=False)

print("\n📌 IMPORTÂNCIA DAS VARIÁVEIS")
print(importances)










# XGBoost

modelo_xgb = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

modelo_xgb.fit(X_treino, y_treino)

pred_xgb = modelo_xgb.predict(X_teste)



print("\n🚀 XGBOOST")

print("R2:", r2_score(y_teste, pred_xgb))

print("MAE:", mean_absolute_error(y_teste, pred_xgb))



importances_xgb = pd.DataFrame({
    "Feature": X.columns,
    "Importancia": modelo_xgb.feature_importances_
}).sort_values(by="Importancia", ascending=False)

print(importances_xgb)





plt.figure(figsize=(8,5))

plt.barh(
    importances_xgb["Feature"],
    importances_xgb["Importancia"]
)

plt.xlabel("Importância")
plt.title("Importância das Variáveis - XGBoost")

plt.show()




plt.scatter(y_teste, pred_xgb)

plt.xlabel("Valores Reais")
plt.ylabel("Previsões XGBoost")

plt.title("Real vs Previsto - XGBoost")

plt.show()





from sklearn.metrics import root_mean_squared_error

rmse_xgb = root_mean_squared_error(
    y_teste,
    pred_xgb
)

print("RMSE:", rmse_xgb)


# salvar o melhor modelo (XGBoost)
joblib.dump(modelo_xgb, "modelo_xgb.pkl")

print("Modelo salvo com sucesso!")