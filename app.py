import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report


def main():
    model = load_model("model_dumps/model.pkl")
    test_data = load_test_data("data/preprocessed_data.csv")

    y = test_data['hazardous']
    X = test_data.drop(['hazardous'], axis=1)

    page = st.sidebar.selectbox(
        "Выберите страницу",
        ["Описание задачи и данных", "Запрос к модели"]
    )

    if page == "Описание задачи и данных":
        st.title("Описание задачи и данных")
        st.write("Выберите страницу слева")

        st.header("Описание задачи")
        st.markdown("""
        Набор данных содержит информацию о сертифицированных НАСА астероидах, вращающихся вокруг Земли.\n
        Цель этой модели - предсказать, является ли конкретный астероид опасным для Земли или нет, на основе определенных параметров, доступных в наборе данных.
        """)

        st.header("Описание данных")
        st.markdown("""Предоставленные данные:\n
вещественные признаки:
* est_diameter_max - размер объекта
* relative_velocity - скорость объекта относительно Земли
* miss_distance - дистанция пролета в километрах
* absolute_magnitude - абсолютная звёздная величина\n
бинарный признак:
* hazardous - показывает, является ли астероид опасным или нет""")

    elif page == "Запрос к модели":
        st.title("Запрос к модели")
        st.write("Выберите страницу слева")
        request = st.selectbox(
            "Выберите запрос",
            ["Сделать прогноз", "Метрики", "Первые 20 предсказанных значений"]
        )

        if request == "Метрики":
            st.header("Метрики")
            y_pred = model.predict(X)
            cr = classification_report(y, y_pred)
            st.write(cr)
            # 'Classification Report: ',cr
            st.write(confusion_matrix(y, y_pred))
        elif request == "Первые 20 предсказанных значений":
            st.header("Первые 20 предсказанных значений")
            y_pred = model.predict(X.iloc[:20, :])
            for item in y_pred:
                st.write(f"{item:.2f}")
        elif request == "Сделать прогноз":
            st.header("Сделать прогноз")

            est_diameter_max = st.number_input("est_diameter_max", 0., 100.)
            relative_velocity = st.number_input("relative_velocity", 100., 300000.)
            miss_distance = st.number_input("miss_distance", 5e+2, 8e+07)
            absolute_magnitude = st.number_input("absolute_magnitude", 0., 50.)

            if st.button('Предсказать'):
                data = [est_diameter_max, relative_velocity, miss_distance, absolute_magnitude]
                data = np.array(data).reshape((1, -1))
                pred = model.predict(data)

                if pred[0]:
                    st.write("Возможна угроза")
                else:
                    st.write("Не представляет опасности")
            else:
                pass


@st.cache_data
def load_model(path_to_file):
    with open(path_to_file, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


@st.cache_data
def load_test_data(path_to_file):
    df = pd.read_csv(path_to_file, sep=";")
    return df


if __name__ == "__main__":
    main()
