import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Input
from keras.src.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime
from io import BytesIO
import os
import yfinance as yf

from chart_utils import *

WINDOW = 20
PERCENTAGE = 0.8

#собираем датасет 
start_date = "2016-02-01"
end_date = "2026-04-17"

tickers = {
    "USD_RUB": "RUB=X",   # курс доллара к рублю
    "Gold": "GC=F",       # золото (USD за унцию)
    "Brent": "BZ=F"       # нефть Brent (USD за баррель)
}


def load_usd_rub_cbr(start_date, end_date):
    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"

    params = {
        "date_req1": datetime.strptime(start_date, "%Y-%m-%d").strftime("%d/%m/%Y"),
        "date_req2": datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y"),
        "VAL_NM_RQ": "R01235"  # USD
    }

    response = requests.get(url, params=params)
    data = pd.read_xml(BytesIO(response.content))

    data = data[["Date", "Value"]]
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True)

    # заменяем запятые
    data["Value"] = data["Value"].str.replace(",", ".").astype(float)

    data = data.set_index("Date")
    data = data.rename(columns={"Value": "USD_RUB"})

    return data


def load_dataset():
    if not os.path.exists("dataset_prepared.csv"):

        print("Скачиваем данные...")

        # --- USD/RUB с ЦБ ---
        usd = load_usd_rub_cbr(start_date, end_date)

        # --- золото и нефть ---
        raw = yf.download(
            ["GC=F", "BZ=F"],
            start=start_date,
            end=end_date
        )["Close"]

        raw.columns = ["Gold", "Brent"]

        # --- объединение ---
        data = usd.join(raw, how="inner")

        # --- только рабочие дни ---
        data = data.asfreq("B")

        # --- заполнение пропусков ---
        data = data.ffill().dropna()

        print(data.tail())

        # --- returns ---
        returns = data.pct_change().dropna()
        returns.columns = [c + "_return" for c in returns.columns]

        final_data = pd.concat([data, returns], axis=1).dropna()

        final_data.to_csv("dataset_prepared.csv")

        print("Датасет сохранён")

    else:
        print("Используем готовый датасет")

    return pd.read_csv("dataset_prepared.csv", index_col=0)

 
#многомерный
def create_prepared_arrays(data, target_columns, start_index, WINDOW, PERCENTAGE):
    """
    Более гибкая версия с шагом и индексом начала
    Параметры:
    -----------
    dataset : numpy.ndarray
        Массив всех признаков
    
    target_columns : list
        Индексы колонок для прогнозирования (два параметра)
    
    start_index : int
        Начальный индекс для создания последовательностей
    
    history_size : int
        Размер истории (длина окна)
    
    step : int
        Шаг между элементами в последовательности (1 = каждые, 2 = каждый второй)
    
    train_ratio : float
        Доля обучающей выборки
    """
    
    data = []
    labels = []
    
    # Извлекаем целевые признаки
    targets = data[:, target_columns]
    
    end_index = start_index + WINDOW
    
    for i in range(start_index, end_index):
        # Создаем последовательность с шагом step
        indices = range(i, i + WINDOW) 
        data.append(data[indices])
        labels.append(targets[i + WINDOW])
    
    X = np.array(data)
    Y = np.array(labels)
    
    # Разделение на train/test
    train_size = int(len(X) * PERCENTAGE)
    
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]
    
    return X_train, Y_train, X_test, Y_test


def prepare_data(features_cols):
    data = pd.read_csv("dataset_prepared.csv", index_col=0)

    # модель 1
    features = data[features_cols]
    
    # модель 3
    returns = data[["USD_RUB", "Gold", "Brent"]].pct_change().dropna()

    # нормализация
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    # создаем последовательности
    X, Y = [], []

    for i in range(len(scaled) - WINDOW):
        X.append(scaled[i:i+WINDOW])
        Y.append(scaled[i+WINDOW][0])  # прогнозируем USD_RUB

    X = np.array(X)
    Y = np.array(Y)

    # train/test split
    split = int(len(X) * PERCENTAGE)

    x_train = X[:split]
    y_train = Y[:split]
    x_test = X[split:]
    y_test = Y[split:]

    return x_train, x_test, y_train, y_test, scaler

# espeshealy for model 3
def prepare_data_returns():
    data = pd.read_csv("dataset_prepared.csv", index_col=0)

    returns = data[["USD_RUB", "Gold", "Brent"]].pct_change().dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(returns)

    X, y = [], []

    for i in range(len(scaled) - WINDOW):
        X.append(scaled[i:i+WINDOW])
        y.append(scaled[i+WINDOW][0])  # доходность USD/RUB

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * PERCENTAGE)

    x_train = X[:split]
    y_train = y[:split]
    x_test = X[split:]
    y_test = y[split:]

    return x_train, x_test, y_train, y_test, scaler


def create_and_train_model(x_train, y_train, x_test, y_test, model_name):

    if os.path.exists(model_name):
        print(f"Загружаем модель {model_name}")
        return tf.keras.models.load_model(model_name)

    print(f"Обучаем модель {model_name}")

    model = Sequential()
    model.add(Input(shape=(WINDOW, x_train.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(1))
    

    model.compile(loss="mse", optimizer=Adam())

    history = model.fit(
        x_train, y_train,
        epochs=150,
        batch_size=16,
        validation_data=(x_test, y_test),
        verbose=2
    )

    model.save(model_name)

    print_metrics_charts(history)
    print_text_results(model, x_test, y_test, x_train, y_train)

    return model

def returns_to_price(pred_returns, last_price):
    prices = []
    current_price = last_price

    for r in pred_returns:
        current_price = current_price * (1 + r)
        prices.append(current_price)

    return np.array(prices)

def predict(model, x_train, y_train, x_test, y_test, scaler, model_name):
    train_predict = model.predict(x_train)  # генерирует прогноз для входной выборки.
    test_predict = model.predict(x_test)
    
    # перевод в рубли
    train_predict_real = inverse_transform(scaler, train_predict.flatten())
    y_train_real = inverse_transform(scaler, y_train.flatten())

    test_predict_real = inverse_transform(scaler, test_predict.flatten())
    y_test_real = inverse_transform(scaler, y_test.flatten())
    
    print(f"\n---- {model_name} ----")
    
    # пользовательский метод отрисовки графика прогнозирования
    print_forecast_charts(
        test_predict_real,
        y_test_real,
        train_predict_real,
        y_train_real,
        title=f"{model_name} (RUB)"
    )

    # новый график только теста
    plot_test_only(y_test_real, test_predict_real)
    
    # Дополнительно для интерпретации результатов рассчитывались метрики RMSE и MAE
    print_metrics_real(y_test_real, test_predict_real)

    # последнее значение
    last_pred = test_predict_real[-1]
    print(f"\nПоследний прогноз (USD/RUB): {last_pred:.2f}")

    return last_pred


def predict_returns(model, x_test, y_test, scaler, data_full):

    pred_scaled = model.predict(x_test).flatten()

    # возвращаем реальные returns
    pred_returns = inverse_transform(scaler, pred_scaled)
    y_returns = inverse_transform(scaler, y_test)

    print("\n---- Returns model ----")

    # график В RETURNS
    plot_returns(y_returns, pred_returns)

    # метрики В RETURNS
    print_metrics_returns(y_returns, pred_returns)

    # прогноз на завтра (единственное место, где переводим в рубли)
    last_price = data_full["USD_RUB"].iloc[-1]
    next_return = pred_returns[-1]

    next_price = last_price * (1 + next_return)

    print(f"\nПрогноз на завтра (returns): {next_price:.2f} RUB")

    return next_price

def predict_next_day_levels(model, data, scaler, feature_cols):

    features = data[feature_cols]

    scaled = scaler.transform(features)

    last_window = scaled[-WINDOW:]
    last_window = np.expand_dims(last_window, axis=0)

    pred = model.predict(last_window)

    pred_real = inverse_transform(scaler, pred.flatten())[0]

    print(f"Прогноз на завтра ({feature_cols}): {pred_real:.2f} RUB")

    return pred_real


def predict_next_day_returns(model, data, scaler):

    returns = data[["USD_RUB", "Gold", "Brent"]].pct_change().dropna()

    scaled = scaler.transform(returns)

    last_window = scaled[-WINDOW:]
    last_window = np.expand_dims(last_window, axis=0)

    pred_scaled = model.predict(last_window)[0][0]

    pred_return = inverse_transform(scaler, np.array([pred_scaled]))[0]

    last_price = data["USD_RUB"].iloc[-1]

    next_price = last_price * (1 + pred_return)

    print(f"Прогноз на завтра (returns): {next_price:.2f} RUB")

    return next_price


if __name__ == '__main__':

    data_full = load_dataset()

    # ===== GOLD =====
    # x_train_g, x_test_g, y_train_g, y_test_g, scaler_g = prepare_data(["USD_RUB", "Gold"])

    # model_g = create_and_train_model(
    #     x_train_g, y_train_g, x_test_g, y_test_g,
    #     "model_gold.keras"
    # )

    # predict(model_g, x_train_g, y_train_g, x_test_g, y_test_g, scaler_g, "MODEL GOLD")

    # predict_next_day_levels(model_g, data_full, scaler_g, ["USD_RUB", "Gold"])


    # # ===== OIL =====
    # x_train_o, x_test_o, y_train_o, y_test_o, scaler_o = prepare_data(["USD_RUB", "Brent"])

    # model_o = create_and_train_model(
    #     x_train_o, y_train_o, x_test_o, y_test_o,
    #     "model_oil.keras"
    # )

    # predict(model_o, x_train_o, y_train_o, x_test_o, y_test_o, scaler_o, "MODEL OIL")

    # predict_next_day_levels(model_o, data_full, scaler_o, ["USD_RUB", "Brent"])


    # ===== RETURNS =====
    x_train_r, x_test_r, y_train_r, y_test_r, scaler_r = prepare_data_returns()

    model_r = create_and_train_model(
        x_train_r, y_train_r, x_test_r, y_test_r,
        "model_returns.keras"
    )

    predict_returns(model_r, x_test_r, y_test_r, scaler_r, data_full)

    predict_next_day_returns(model_r, data_full, scaler_r)