import pickle

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from utils.constants import MODEL_PATH, DF_TRAIN_URL
from utils.preprocessors import ColumnConverter, get_object, get_float


@st.cache_resource
def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Пикл файл с моделью отсутствует")

    with MODEL_PATH.open("rb") as f:
        model: Pipeline = pickle.load(f)

    return model


@st.cache_data
def load_dataframe(src: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(src)
    except Exception:
        raise ValueError("Указана неправильный путь до датафрейма")

    return df


@st.cache_data
def get_model_info(_model: Pipeline) -> dict[str, Any]:
    model_class: Ridge = _model["regressor"]
    converter: ColumnTransformer = _model["converter"]
    preprocessor: ColumnTransformer = _model["preprocessor"]

    df_train = load_dataframe(DF_TRAIN_URL)
    columns_without_target = df_train.drop(["selling_price"], axis=1).columns
    df_train = df_train.drop_duplicates(columns_without_target, ignore_index=True)

    num_objects = df_train.shape[0]

    X_train = df_train.drop("selling_price", axis=1)
    y_train = df_train["selling_price"]

    predictions = _model.predict(X_train)

    coefs = model_class.coef_
    coefs_names = preprocessor.transform(converter.transform(X_train)).columns
    weights = pd.DataFrame({"Название": coefs_names, "Значение": coefs})
    r2 = r2_score(y_train, predictions)

    return {
        "class": model_class,
        "weights": weights,
        "num_objects": num_objects,
        "r2": r2
    }


def show_dataset_info(df: pd.DataFrame) -> None:
    st.subheader("Первые 5 строк")
    st.dataframe(df.head())

    st.subheader("Информация по числовым столбцам")
    st.dataframe(df.describe())

    st.subheader("Информация по категориальным столбцам")
    st.dataframe(df.describe(include="object"))

    st.subheader("Количество пропусков")
    st.dataframe(
        pd.DataFrame(
            df.isna().sum(),
            columns=["Количество пропусков"]).reset_index().rename(columns={"index": "Название столбца"}
        ),
        width="content",
    )

    st.subheader("Количество дубликатов")
    if "selling_price" in df.columns:
        columns_for_showing = df.drop(["selling_price"], axis=1).columns
    else:
        columns_for_showing = df.columns
    st.write(df.duplicated(columns_for_showing).sum())

    if "selling_price" in df.columns:
        st.subheader("Распределение целевой переменной")
        fig, ax = plt.subplots()
        sns.histplot(df["selling_price"], kde=True, ax=ax)
        st.pyplot(fig, width=800)
    else:
        st.subheader("Целевая переменная отсутствует в датасете")

    st.subheader("Тепловая карта корреляции")
    df_mod = prepare_dataset_via_model(df)

    corr = df_mod.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm'
    )
    st.pyplot(fig, width=800)


def prepare_dataset_via_model(df: pd.DataFrame, model: Pipeline | None = None) -> pd.DataFrame:
    if model is None:
        model = load_model()

    converter: ColumnTransformer = model["converter"]

    return converter.transform(df)
