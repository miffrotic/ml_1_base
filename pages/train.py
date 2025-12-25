import streamlit as st

from utils.common import load_dataframe, show_dataset_info
from utils.constants import DF_TRAIN_URL


st.set_page_config(page_title="Тренировочный датасет", layout="wide")
st.title("Тренировочный датасет")

df = load_dataframe(DF_TRAIN_URL)
show_dataset_info(df)
