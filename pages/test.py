import streamlit as st

from utils.common import load_dataframe, show_dataset_info
from utils.constants import DF_TEST_URL


st.set_page_config(page_title="Тестовый датасет", layout="wide")
st.title("Тестовый датасет")

df = load_dataframe(DF_TEST_URL)
show_dataset_info(df)
