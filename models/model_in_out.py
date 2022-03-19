
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import charts


class ModelInOut:
    """Класс для построения модели распознавания
    входов выходов и удержания позиции отдельного токена"""

    def __init__(self, token: str, data: ()) -> None:
        self.token = token
        self.x_train = data[0]
        self.x_test = data[1]
        self.y_train = data[2]
        self.y_test = data[3]
        self.prices = data[4]
        self.indexes = data[5]

        # Построим график по результатам обучения модели
        self.chart_builder = charts.LearningChartBuilder()

        # Строим архитектуру для модели
        self._build_model()

        # Начинаем обучать модель
        self._train_model()

    def _build_model(self) -> None:
        """Метод для построения архитектуры модели"""
        input = Input(shape=(10,))

        x = Dense(600, activation='relu')(input)
        x = Dropout(0.3)(x)
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(50, activation='sigmoid')(x)

        output = Dense(3, activation='softmax')(x)

        self.model = Model(input, output)
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    def _train_model(self) -> None:
        """Метод для тренировки модели"""
        # history = self.model.fit(
        #     self.x_train[: - 1500 - 10 - 2],
        #     self.y_train[: - 1500 - 10 - 2],
        #     batch_size=64,
        #     epochs=20,
        #     validation_data=[self.x_train[-1500:], self.y_train[-1500:]],
        #     verbose=1,
        #     shuffle=True
        # )
        # # Кинем данные по обучению в график и построим его
        # self.chart_builder.set_history(history.history)
        # self.chart_builder.build()
        # self._evaluate_model()
        # self.model.save_weights("my_checkpoint")

        self.model.load_weights("my_checkpoint")

    def _evaluate_model(self) -> None:
        """Метод для проверки модели на тестовых данных"""
        stats = self.model.evaluate(self.x_test, self.y_test)
        print('Evaluate:', stats)

    def predict(self) -> None:
        """Метод для прогноза цены"""
        predictions = self.model.predict(self.x_test)
        true_dim = self.y_test

        for index, prediction in enumerate(predictions):
            print("Прогноз: {A} Правильное: {B}" . format(A=prediction, B=true_dim[index]))

        signs = []
        for index, signal in enumerate(predictions):
            signs.append(np.argmax(signal))
        signs = np.array(signs)

        t_signs = []
        for index, signal in enumerate(true_dim):
            t_signs.append(np.argmax(signal))
        t_signs = np.array(t_signs)

        print("Total preds",
              predictions.shape,
              "Total hold",
              len(signs[signs == 0]),
              "Total sell",
              len(signs[signs == 1]),
              "Total buy",
              len(signs[signs == 2]))
        print("true:")
        print("Total preds",
              true_dim.shape,
              "Total hold",
              len(t_signs[t_signs == 0]),
              "Total sell",
              len(t_signs[t_signs == 1]),
              "Total buy",
              len(t_signs[t_signs == 2]))
        # print(len(self.prices))
        # print(234324, self.prices[29798:])

        self.trade_builder = TradeChartBuilder(self.token, self.prices, self.indexes, predictions, true_dim)
        self.trade_builder.build()

        # plt.style.use("seaborn-dark")
        #
        # for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        #     plt.rcParams[param] = '#212946'
        # for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        #     plt.rcParams[param] = '0.9'
        #
        # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        #
        # ax.grid(color='#2A3459')
        #
        # x_buy_signals = [signal['Start'] for signal in signals if signal['Type'] == 'BUY']
        # x_sell_signals = [signal['Start'] for signal in signals if signal['Type'] == 'SELL']
        #
        # ax.plot(self.prices[29798:, 1].astype(float), label='BTC USDT')
        # ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        # ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        #
        # plt.legend()
        # plt.show()

    def stats(self) -> None:
        """Метод для вывода статистики тренировки модели"""
        pass

# data = yf.download("BTC-USD", interval="5m", start="2020-01-01", end="2020-03-01")
# print(data)
