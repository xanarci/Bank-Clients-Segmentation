import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Bank Client Segmentation', layout='centered')


@st.cache_resource
def load_artifacts():
    kmeans = joblib.load('kmeans_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # cluster descriptions
    try:
        cluster_profile = pd.read_csv('cluster_profile.csv', index_col=0)
    except Exception:
        cluster_profile = None
    return kmeans, scaler, cluster_profile


kmeans, scaler, cluster_profile = load_artifacts()

st.title('Сегментация клиентов банка — KMeans')
st.write('Введите параметры клиента, чтобы отнести его к сегменту.')

# Список признаков (убедись, что совпадает с обучением)
feature_names = [
    'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES',
    'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY',
    'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
    'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX',
    'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
]

# Поля для ввода данных
user_input = {}
for feat in feature_names:
    user_input[feat] = st.number_input(feat, value=0.0)

if st.button('Классифицировать'):
    X_new = pd.DataFrame([user_input])
    X_scaled = scaler.transform(X_new)
    cluster = kmeans.predict(X_scaled)[0]
    st.success(f'Клиент отнесён к сегменту: {cluster}')

    if cluster_profile is not None:
        st.subheader('Профиль сегмента (средние значения признаков)')
        if str(cluster) in cluster_profile.columns:
            st.dataframe(cluster_profile[[str(cluster)]])
        else:
            st.dataframe(cluster_profile)
    else:
        st.write('Профиль сегмента не найден (cluster_profile.csv отсутствует).')

    st.subheader('Рекомендации')
    st.write('- Пример: Клиент из сегмента X — предложить премиальную карту с кешбэком.')