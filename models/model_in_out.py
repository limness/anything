import yfinance as yf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TradeChartBuilder:
    """Класс для построения графиков
    результатов торгов на бирже"""

    def __init__(self, token: str, prices: [], indexes: [], predictions: [], true_dim: []) -> None:
        self.prices = prices
        self.predictions = predictions
        self.indexes = indexes
        self.true_dim = true_dim
        self.token = token

    def build(self) -> None:
        """Метод для вывода статистики трейдинга"""
        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.grid(color='#2A3459')

        self._draw_price_history(ax)

        plt.legend()
        plt.show()

    def _draw_price_history(self, ax: plt) -> None:
        """A method that draws profit graphic using a strategy"""

        preds = np.array(self.predictions)

        # print('BUILD CHART')
        x_buy_signals = [self.indexes[index] for index, pred in enumerate(preds) if np.argmax(pred) == 2]
        x_sell_signals = [self.indexes[index] for index, pred in enumerate(preds) if np.argmax(pred) == 1]
        # x_hold_signals = [signal['Start'] for signal in signals if signal['Type'] == 'HOLD']
        #
        y_buy_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(preds) if np.argmax(pred) == 2]
        y_sell_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(preds) if np.argmax(pred) == 1]
        # y_hold_signals = [signal['Price'] for signal in signals if signal['Type'] == 'HOLD']

        ax.plot(self.prices, label="BTCUSDT")
        ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        # ax.scatter(x_hold_signals, y_hold_signals, label="HOLD", s=10, c='orange')


class SignalsChartBuilder:
    """Класс для построения графиков
    сигналов для выхода в сеть"""

    def __init__(self, token: str, norm_data: [], signals: []) -> None:
        self.norm_data = norm_data
        self.signals = signals
        self.token = token

    def build(self) -> None:
        """Метод для вывода статистики трейдинга"""
        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))

        ax0.grid(color='#2A3459')
        ax1.grid(color='#2A3459')

        self._draw_price_history(self.token, self.norm_data, ax0)
        self._draw_signals_history(self.token, self.norm_data, self.signals, ax1)

        plt.legend()
        plt.show()

    def _draw_price_history(self, token: str, prices: [], ax: plt) -> None:
        """A method that draws profit graphic using a strategy"""

        ax.plot(prices[:, 1].astype(float), label=token)

    def _draw_signals_history(self, token: str, prices: [], signals: [], ax: plt) -> None:
        """A method that draws profit graphic using a strategy"""

        x_buy_signals = [signal['Start'] for signal in signals if signal['Type'] == 'BUY']
        x_sell_signals = [signal['Start'] for signal in signals if signal['Type'] == 'SELL']
        x_hold_signals = [signal['Start'] for signal in signals if signal['Type'] == 'HOLD']

        y_buy_signals = [signal['Price'] for signal in signals if signal['Type'] == 'BUY']
        y_sell_signals = [signal['Price'] for signal in signals if signal['Type'] == 'SELL']
        y_hold_signals = [signal['Price'] for signal in signals if signal['Type'] == 'HOLD']

        ax.plot(prices[:, 1].astype(float), label=token)
        ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        ax.scatter(x_hold_signals, y_hold_signals, label="HOLD", s=10, c='orange')

        ax.legend()


class LearningChartBuilder:
    """Класс для построения графиков
    результатов обучения сети"""

    def __init__(self) -> None:
        self.history = {}

    def set_history(self, history) -> None:
        """Метод для загрузки данных обучения"""
        self.history = history

    def build(self) -> None:
        """Метод для вывода статистики трейдинга"""
        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))

        ax0.grid(color='#2A3459')
        ax1.grid(color='#2A3459')

        ax0.plot(self.history['loss'], label="Loss")
        ax0.plot(self.history['val_loss'], label="Val Loss")

        ax1.plot(self.history['accuracy'], label="Accuracy")
        ax1.plot(self.history['val_accuracy'], label="Val Accuracy")

        plt.legend()
        plt.show()


class ModelDataBuilder:
    """Класс для подготовки тренировочных данных для модели"""

    def __init__(self, token: str) -> None:
        self.token = token

        # Получим все необходимые тренировочные данные из файла
        self.data = self._read_token_from_file()[:15000]

        self.x = []
        self.y = []

    def _read_token_from_file(self) -> []:
        """Метод для чтения данных с биржи из файла"""
        with open(f'tokens/{self.token}.pickle', 'rb') as handle:
            klines_per_day = pickle.load(handle)
        return klines_per_day

    def get_signals(self, percent: float, price_history: []) -> []:
        """Метод для получения экстремумов функции в процентах"""
        signals = []

        for index, price in enumerate(price_history):
            if index != 0:
                # Если N-ая цена в графике больше на X% последней точки
                if price > signals[-1]['Price'] + signals[-1]['Price'] / 100 * percent:
                    signals[-1]['Type'] = 'BUY'
                # Если N-ая цена в графике меньше на X% последней точки
                elif price < signals[-1]['Price'] - signals[-1]['Price'] / 100 * percent:
                    signals[-1]['Type'] = 'SELL'
                else:
                    signals[-1]['History'].append(price)
                    continue

            signals.append({
                'Type': 'HZ',
                'Start': index,
                'Price': price,
                'History': [price]
            })

        return np.array(signals)

    def _signals_normalization(self, signals: []) -> []:
        """Метод для преобразования шумовых сигнилов в нормальные"""
        norm_signals = []
        preview_signal_type = 'HZ'

        for signal in signals:
            if signal['Type'] != preview_signal_type:
                norm_signals.append({
                    'Type': signal['Type'],
                    'Start': signal['Start'],
                    'Price': signal['Price'],
                })
                preview_signal_type = signal['Type']

        return np.array(norm_signals)

    def build(self) -> ():
        """Метод для чтения данных с биржи из файла"""

        # Получаем данные по открытиям свечи
        prices = [float(candle[1]) for candle in self.data[:]]

        # Получаем все сигналы по функции
        self.signals = self.get_signals(percent=1.8, price_history=prices)

        self.x = []
        self.y = []

        old_train_x = 10

        self.data = np.array(self.data)

        open = self.data[:, 1].astype(float)
        open = preprocessing.scale(open)

        self.signals = self._signals_normalization(self.signals)

        new_signals = [{'Type': 'HOLD', 'Start': index, 'Price': price} for index, price in enumerate(prices)]

        for index_main, signal in enumerate(self.signals):
            new_signals[signal['Start']]['Type'] = signal['Type']

        # for signal in new_signals:
        #     print(signal)

        self.signals = new_signals
        # for signal in self.signals:
        #     print(signal)

        # print(self.signals)
        # print(self.signals[:])

        # for index in enumerate(prices):
        #     if not index in self.signals[:, 'Start']:
        #         print('sueta')

        # Построим график по входным данным обучения
        # self.trade_builder = TradeChartBuilder(self.token, self.data, self.signals)
        # self.trade_builder.build()

        print('LEN', len([item for item in self.signals if item['Type'] == 'HOLD']))
        print('LEN', len([item for item in self.signals if item['Type'] == 'BUY']))
        print('LEN', len([item for item in self.signals if item['Type'] == 'SELL']))

        # Делаем up sampling, расширение данных
        # Иначе сеть проглотит слишком неравномерное количество равных данных
        len_max = len([item for item in self.signals if item['Type'] == 'HOLD'])

        new_arr_sell = [item for item in self.signals if item['Type'] == 'SELL']
        new_arr_sell *= len_max // len(new_arr_sell)

        new_arr_buy = [item for item in self.signals if item['Type'] == 'BUY']
        new_arr_buy *= len_max // len(new_arr_buy)

        self.signals.extend(new_arr_buy)
        self.signals.extend(new_arr_sell)

        # print('NEW ARR', len(self.signals))
        # print(new_arr_buy)

        indexes = []

        for signal in self.signals:
            if signal['Start'] <= old_train_x:
                continue

            if signal['Type'] == 'BUY':
                y_binary = [0, 0, 1]
            elif signal['Type'] == 'SELL':
                y_binary = [0, 1, 0]
            else:
                y_binary = [1, 0, 0]

            # Входными данными для X являются данные по свечам
            # с момента сигнала и все 5 предыдущих свечей
            self.x.append(open[signal['Start'] - old_train_x:signal['Start']])
            self.y.append(y_binary)
            indexes.append(signal['Start'])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

        # print('self.x', len(self.x))

        # for y in self.y:
        #     print(y)

        indexes_train, \
        indexes_test, \
        x_train, \
        x_test, \
        y_train, \
        y_test = train_test_split(
            indexes,
            self.x,
            self.y,
            test_size=0.33,
            shuffle=True
        )
        print('indexes_test', indexes_test)
        return x_train, x_test, y_train, y_test, prices, indexes_test
        # print(self.x.shape, self.y.shape)


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
        self.chart_builder = LearningChartBuilder()

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
        #     self.x_train,
        #     self.y_train,
        #     batch_size=64,
        #     epochs=100,
        #     validation_split=0.2,
        #     verbose=1,
        #     shuffle=True
        # )
        # # Кинем данные по обучению в график и построим его
        # self.chart_builder.set_history(history.history)
        # self.chart_builder.build()
        self.model.load_weights("my_checkpoint")

    def _evaluate_model(self) -> None:
        """Метод для проверки модели на тестовых данных"""
        pass

    def predict(self) -> None:
        """Метод для прогноза цены"""
        predictions = self.model.predict(self.x_test)
        true_dim = self.y_test

        # for index, prediction in enumerate(predictions):
        #     print("Прогноз: {A} Правильное: {B}" . format(A=prediction, B=true_dim[index]))

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
