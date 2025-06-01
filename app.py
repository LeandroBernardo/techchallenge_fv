import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models():
    try:
        df = pd.read_csv('insurance.csv')
    except FileNotFoundError:
        st.error("Erro: Arquivo 'insurance.csv' não encontrado no diretório.")
        return None, None, None, None, None, None, None, None, 0, 0, 0, 0, 0, 0, 0, 0
    
    df = df.drop_duplicates()
    
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    df['sex'] = le_sex.fit_transform(df['sex'])
    df['smoker'] = le_smoker.fit_transform(df['smoker'])
    df['region'] = le_region.fit_transform(df['region'])

    X = df.drop('charges', axis=1)
    y = df['charges']
    
    scaler = StandardScaler()
    X[['age', 'bmi', 'children']] = scaler.fit_transform(X[['age', 'bmi', 'children']])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LinearRegression()
    dt = DecisionTreeRegressor(random_state=42)
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    knn = KNeighborsRegressor(n_neighbors=5)
    
    lr.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return rmse, r2
    
    lr_rmse, lr_r2 = evaluate_model(lr, X_test, y_test)
    dt_rmse, dt_r2 = evaluate_model(dt, X_test, y_test)
    rf_rmse, rf_r2 = evaluate_model(rf, X_test, y_test)
    knn_rmse, knn_r2 = evaluate_model(knn, X_test, y_test)
    
    return lr, dt, rf, knn, scaler, le_sex, le_smoker, le_region, lr_rmse, lr_r2, dt_rmse, dt_r2, rf_rmse, rf_r2, knn_rmse, knn_r2

# Atualizar a chamada para incluir o KNN
lr, dt, rf, knn, scaler, le_sex, le_smoker, le_region, lr_rmse, lr_r2, dt_rmse, dt_r2, rf_rmse, rf_r2, knn_rmse, knn_r2 = train_models()

if lr is None:
    st.stop()

st.title("Previsão de Custos de Seguro de Saúde")
st.write("Insira os dados abaixo para prever o custo do seguro usando diferentes modelos de Machine Learning.")

age = st.slider("Idade", min_value=18, max_value=80, value=20)
sex = st.selectbox("Sexo", options=['female', 'male'])
bmi = st.number_input("IMC (Body Mass Index)", min_value=10.0, max_value=50.0, value=25.0)
children = st.slider("Número de Filhos", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Fumante", options=['no', 'yes'])
region = st.selectbox("Região", options=['southwest', 'southeast', 'northwest', 'northeast'])

if st.button("Prever Custo do Seguro"):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    input_data['sex'] = le_sex.transform(input_data['sex'])
    input_data['smoker'] = le_smoker.transform(input_data['smoker'])
    input_data['region'] = le_region.transform(input_data['region'])
    
    input_data[['age', 'bmi', 'children']] = scaler.transform(input_data[['age', 'bmi', 'children']])
    
    lr_pred = lr.predict(input_data)[0]
    dt_pred = dt.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]
    knn_pred = knn.predict(input_data)[0]
    
    col1, col2 = st.columns([1, 1])  
    
    with col1:
        st.subheader("Resultados da Previsão")
        st.write(f"**Regressão Linear**: ${lr_pred:.2f}")
        st.write(f"**Árvore de Decisão**: ${dt_pred:.2f}")
        st.write(f"**Random Forest**: ${rf_pred:.2f}")
        st.write(f"**K-Nearest Neighbors**: ${knn_pred:.2f}")
    
    st.markdown(
        """
        <style>
        .divider {
            border-left: 2px solid #ccc;
            height: 100%;
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            top: 0;
        }
        </style>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True
    )
    
    with col2:
        st.subheader("Desempenho dos Modelos (Conjunto de Teste)")
        st.write(f"**Regressão Linear** - RMSE: {lr_rmse:.2f}, R²: {lr_r2:.4f}")
        st.write(f"**Árvore de Decisão** - RMSE: {dt_rmse:.2f}, R²: {dt_r2:.4f}")
        st.write(f"**Random Forest** - RMSE: {rf_rmse:.2f}, R²: {rf_r2:.4f}")
        st.write(f"**K-Nearest Neighbors** - RMSE: {knn_rmse:.2f}, R²: {knn_r2:.4f}")