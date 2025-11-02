# streamlit_app.py
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


# Здесь используйте список признаков, соответствующий датафрейму (num_cols)
# Ниже — пример на основе часто встречающихся колонок в датасете ccdata
feature_names = [
'BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES','INSTALLMENTS_PURCHASES',
'CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT',
'PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE'
]


# Создаем виджеты ввода динамически (подберите диапазоны/шаги разумно)
user_input = {}
for feat in feature_names:
# default value: 0
user_input[feat] = st.number_input(feat, value=0.0)


if st.button('Классифицировать'):
X_new = pd.DataFrame([user_input])
# Проверка порядка колонок: если scaler ожидает другой набор признаков — нужно согласовать
# Применяем scaler
X_scaled = scaler.transform(X_new)
cluster = kmeans.predict(X_scaled)[0]
st.success(f'Клиент отнесён к сегменту: {cluster}')


if cluster_profile is not None:
st.subheader('Профиль сегмента (средние значения признаков)')
st.dataframe(cluster_profile[f'cluster_{cluster}'] if f'cluster_{cluster}' in cluster_profile.columns else cluster_profile.loc[:,str(cluster)])
else:
st.write('Профиль сегмента сохранён не был — откройте cluster_profile.csv для подробностей.')


# Место для бизнес-рекомендаций
st.subheader('Рекомендации')
st.write('- Пример: Клиент из сегмента X — предложить кредитную карту с кешбэком')