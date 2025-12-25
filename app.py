import random

import numpy as np
import streamlit as st

from utils.constants import PAGES_PATH
from utils.preprocessors import ColumnConverter, get_object, get_float


random.seed(42)
np.random.seed(42)

pages = [
    st.Page(PAGES_PATH / "info.py", title="Обученная модель"),
    st.Page(PAGES_PATH / "train.py", title="Тренировочный датасет"),
    st.Page(PAGES_PATH / "test.py", title="Тестовый датасет"),
    st.Page(PAGES_PATH / "load.py", title="Загрузить файл"),
    st.Page(PAGES_PATH / "input.py", title="Ввести данные объекта", default=True),
]

pg = st.navigation(pages, position="sidebar")
pg.run()
