import streamlit as st
import pandas as pd

from utils.common import show_dataset_info, load_model


st.set_page_config(page_title="Предсказание по файлу", layout="wide")
st.title("Загрузить файл CSV")

uploaded_file = st.file_uploader("Выбрать файл", type="csv")
if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)
show_dataset_info(df)

if st.button("Предсказать цену"):
    if "selling_price" in df.columns:
        df = df.drop("selling_price", axis=1)

    model = load_model()

    predictions = model.predict(df)
    result = pd.DataFrame({"Цена": predictions}).round(0)

    st.subheader("Предсказанные цены")
    st.dataframe(result, width=800)
