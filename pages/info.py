import streamlit as st

from utils.common import load_model, get_model_info


st.set_page_config(page_title="Информация о модели", layout="centered")
st.title("Информация о модели")

model = load_model()
model_info = get_model_info(model)

st.subheader(f"Класс модели: {model_info["class"]}")

st.subheader("Веса модели")
st.dataframe(model_info["weights"])

st.subheader(f"Обучено на {model_info["num_objects"]} объектах")
st.subheader(f"Метрика R2: {model_info["r2"]:.5f}")
