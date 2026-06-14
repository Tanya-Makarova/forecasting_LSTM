import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def print_metrics_charts(history):
    """
    Графики обучения модели
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss during training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()


def print_metrics_real(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("\n---- Real metrics (RUB) ----")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    

def print_text_results(model, x_test, y_test, x_train, y_train):
    """
    Вывод метрик модели
    """
    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)

    print('\n---- Model Results ----')

    if isinstance(train_score, list):
        print(f"Train Loss: {train_score[0]:.6f}")
        print(f"Test Loss:  {test_score[0]:.6f}")
    else:
        print(f"Train Loss: {train_score:.6f}")
        print(f"Test Loss:  {test_score:.6f}")


def inverse_transform(scaler, data, feature_index=0):
    """
    Обратное преобразование нормализации (только для одной переменной)

    scaler — MinMaxScaler
    data — массив значений (предсказания)
    feature_index — индекс нужного признака (USD_RUB = 0)
    """
    dummy = np.zeros((len(data), scaler.scale_.shape[0]))
    dummy[:, feature_index] = data
    return scaler.inverse_transform(dummy)[:, feature_index]


def print_forecast_charts(
        test_predict,
        y_test,
        train_predict,
        y_train,
        title="Forecast",
        inverse=False,
        scaler=None
):
    """
    Визуализация прогноза

    учитывает:
    - flatten
    - смещение временного ряда
    - возможность денормализации
    """

    # преобразуем форму
    train_predict = train_predict.flatten()
    test_predict = test_predict.flatten()

    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # денормализация (если нужно)
    if inverse and scaler is not None:
        train_predict = inverse_transform(scaler, train_predict)
        test_predict = inverse_transform(scaler, test_predict)
        y_train = inverse_transform(scaler, y_train)
        y_test = inverse_transform(scaler, y_test)

    # общий график
    plt.figure(figsize=(12, 6))

    # train часть
    plt.plot(y_train, label="train real", alpha=0.7)
    plt.plot(train_predict, label="train predicted", alpha=0.7)

    # test часть со сдвигом
    offset = len(y_train)

    plt.plot(
        range(offset, offset + len(y_test)),
        y_test,
        label="test real",
        alpha=0.7
    )

    plt.plot(
        range(offset, offset + len(test_predict)),
        test_predict,
        label="test predicted",
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


def plot_test_only(y_test, test_predict):
    plt.figure(figsize=(12, 6))

    plt.plot(y_test, label="real")
    plt.plot(test_predict, label="predicted")

    plt.title("Test Forecast Only")
    plt.xlabel("Time")
    plt.ylabel("USD/RUB")
    plt.legend()
    plt.grid()

    plt.show()

def plot_returns(y_true, y_pred):
    plt.figure(figsize=(12, 6))

    plt.plot(y_true, label="real return")
    plt.plot(y_pred, label="predicted return")

    plt.title("Returns Forecast")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.grid()

    plt.show()
    

def print_metrics_returns(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("\n---- Metrics (RETURNS) ----")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")

