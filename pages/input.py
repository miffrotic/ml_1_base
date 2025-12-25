import streamlit as st
import pandas as pd

from utils.common import load_model


st.set_page_config(page_title="Предсказание по объекту", layout="centered")
st.title("Ручной ввод данных")

name = st.text_input("Название (name)", "Mahindra Xylo E4 BS IV")
year = st.number_input("Год выпуска (year)", min_value=1900, max_value=2025, value=2010)
km_driven = st.number_input("Пробег (km_driven)", min_value=0, value=168000)
fuel = st.selectbox("Тип топлива (fuel)", ["Diesel", "Petrol", "LPG", "CNG"])
seller_type = st.selectbox("Тип продавца (seller_type)", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Трансмиссия (transmission)", ["Manual", "Automatic"])
owner = st.selectbox("Тип владельца (owner)", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
mileage = st.number_input("Расход топлива, kmpl (mileage)", min_value=0.0, value=14.0)
engine = st.number_input("Объём двигателя, CC (engine)", min_value=0, value=2498)
max_power = st.number_input("Макс. мощность, bhp (max_power)", min_value=0.0, value=112.0)
torque = st.number_input("Крутящий момент, rpm (torque)", min_value=0, value=2000)
seats = st.number_input("Количество мест (seats)", min_value=2, value=7)

if st.button("Предсказать цену"):
    input_data = pd.DataFrame(
        {
            "name": [name],
            "year": [year],
            "km_driven": [km_driven],
            "fuel": [fuel],
            "seller_type": [seller_type],
            "transmission": [transmission],
            "owner": [owner],
            "mileage": [mileage],
            "engine": [engine],
            "max_power": [max_power],
            "torque": [torque],
            "seats": [seats],
        }
    )

    model = load_model()

    prediction = model.predict(input_data)[0]
    st.success(f"Предсказанная цена: {int(prediction)}")
