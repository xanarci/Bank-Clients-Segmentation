# Bank Client Segmentation 

Проект: сегментация клиентов банков по поведению по кредитной карте.

## Структура репозитория

- `notebook_bank_segmentation.ipynb` — полный анализ и обучение модели (EDA, предобработка, подбор k, обучение, сохранение артефактов).
- `streamlit_app.py` — Streamlit-приложение для классификации нового клиента.
- `kmeans_model.joblib`, `scaler.joblib` — сгенерированные артефакты (не закоммичены в репозиторий, но можно добавить в release).
- `cluster_profile.csv` — профили кластеров (средние значения признаков).
- `requirements.txt` — зависимости.


```bash
pip install -r requirements.txt
