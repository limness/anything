import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
from charts import ChartBuilder


class MarkupBuilder:
    """Класс для расставления разметки для обучения"""

    def __init__(self, data, serializer="pkl") -> None:
        self.data = data
        self.serializer = serializer

    def make_markup(self, percent: float, candles: []) -> []:
        """Метод для получения экстремумов функции в процентах"""
        signals = [
            [145, "BUY"], [183, "SELL"],
            [381, "BUY"], [403, "SELL"],
            [501, "BUY"], [515, "SELL"],
            [617, "BUY"], [729, "SELL"],

            [1615, "BUY"], [1645, "SELL"],
            [1743, "BUY"], [1802, "SELL"],
            [2114, "BUY"], [2119, "SELL"],
            [2171, "BUY"], [2261, "SELL"],

            [2270, "BUY"], [2916, "SELL"],
            [3154, "BUY"], [3224, "SELL"],
        ]
        # plt.plot(candles[:, 0])
        # plt.grid()
        # plt.show()
        # for index, price in enumerate(price_history):
        #     current_price = price
        #     interval = 0
        #     while interval <= max_interval or \
        #             interval + index > len(price_history) - 1:
        #         if price_history[index + interval] > current_price + current_price / 100 * percent:
        #             # Нашли что хотели
        #             pass
        #         elif price_history[index + interval] < current_price - current_price / 100 * percent:
        #             # Нашли что хотели
        #             pass
        # for index, price in enumerate(price_history):
        #     if index != 0:
        #         # Если N-ая цена в графике больше на X% последней точки
        #         if price > signals[-1]['Price'] + signals[-1]['Price'] / 100 * percent:
        #             signals[-1]['Type'] = 'BUY'
        #         # Если N-ая цена в графике меньше на X% последней точки
        #         elif price < signals[-1]['Price'] - signals[-1]['Price'] / 100 * percent:
        #             signals[-1]['Type'] = 'SELL'
        #         else:
        #             signals[-1]['History'].append(price)
        #             continue
        #
        #     signals.append({
        #         'Type': 'HZ',
        #         'Start': index,
        #         'Price': price,
        #         'History': [price]
        #     })

        return np.array(signals)

    # def _signals_normalization(self, signals: []) -> []:
    #     """Метод для преобразования шумовых сигнилов в нормальные"""
    #     norm_signals = []
    #     preview_signal_type = 'HZ'
    #
    #     for signal in signals:
    #         if signal['Type'] != preview_signal_type:
    #             norm_signals.append({
    #                 'Type': signal['Type'],
    #                 'Start': signal['Start'],
    #                 'Price': signal['Price'],
    #             })
    #             preview_signal_type = signal['Type']
    #
    #     return np.array(norm_signals)


class FeaturesBuilder:
    """Класс для добавления фич внутри данных"""

    def __init__(self, data, serializer="pkl") -> None:
        self.data = data
        self.serializer = serializer

    def make_features(self) -> pd.DataFrame:
        pass


class DataBuilder:
    """Класс для подготовки тренировочных данных для модели"""

    def __init__(self, token: str, serializer="csv", show_forward=False,
                 embed_train=False, from_end=True, step_window=500,
                 train_window=1000, val_window=0.4, test_window=300) -> None:
        self.token = token
        self.serializer = serializer
        self.show_forward = show_forward
        self.embed_train = embed_train
        self.step_window = step_window
        self.from_end = from_end

        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.data = self._try_load_or_make_dataset()
        self.scaler = preprocessing.MinMaxScaler()

    def _try_load_or_make_dataset(self) -> pd.DataFrame:
        """Метод для формирования первоначального датасета"""
        try:
            # Попробуем загрузить файл из директории
            data = pd.read_csv(f"datasets/{self.token}_.csv")
        except FileNotFoundError:
            # Файл не найден, необходимо сгенерировать датасет с нуля
            # для этого выгрузим не отформатированные данные из каталога _no_format
            data = self._read_dataset_from_file()[:5000, :-1]
            # Переводим numpy массив в DataFrame, чтобы в дальнейшем было удобно работать с данными
            data = pd.DataFrame(data, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
            data = data.set_index("Datetime")
            # Добавим фичи
            # data = FeaturesBuilder(data).make_features()
            # Сохраним результаты в файл, что в дальнейшем выгружать все из кеша
            data.to_csv(f"datasets/{self.token}_.csv")
        # Сформировать окна для модели
        self._form_windows(data)
        # Если включено отображение форвардного анализа - выводим график
        if self.show_forward:
            print("test_index", self.test_index,
                  "train_index", self.train_index,
                  "val_index", self.val_index)
            ChartBuilder(self.token, data,
                         self.train_index, self.val_index, self.test_index,
                         self.train_window, int(self.train_window * self.val_window), self.test_window).draw()
        return data

    def _form_windows(self, data: pd.DataFrame) -> None:
        """Метод для формирования окон данных"""

        # Тренировочная выборка будет определена как начало датасета + шаг окна
        self.train_index = self.step_window
        # Проверочная выборка будет определена как конец тренировочного окна,
        # если выборка будет не вложенная. Иначе умещаем выборку в тренировочную
        self.val_index = self.train_index + self.train_window \
            if not self.embed_train else self.train_index + self.train_window - int(self.train_window * self.val_window)
        # Тестовая выборка будет определена как конец валидационного окна
        self.test_index = self.val_index + int(self.val_window * self.train_window)

        # Если необходимо брать окно с конца, переносим
        if self.from_end:
            # Отнимем от общего размера данных конец нашего последнего окна
            # и получим общий индекс свдига
            index_offset = data.shape[0] - (self.test_index + self.test_window)

            self.train_index += index_offset
            self.val_index += index_offset
            self.test_index += index_offset

    def _read_dataset_from_file(self) -> np.array:
        """Метод для чтения данных с биржи из файла"""
        with open(f'tokens/{self.token}.pickle', 'rb') as handle:
            klines_per_day = pickle.load(handle)
        return np.array(klines_per_day).astype(float)

    def build(self) -> ():
        """Метод для чтения данных с биржи из файла"""

        # Загрузим данные
        # Определим сигналы
        # Определим фичи
        # Нормализуем данные
        # Сплит данных

        self.get_signals(percent=0.8, candles=self.data)
        # Получаем данные по открытиям свечи
        # prices = [float(candle[1]) for candle in self.data[:]]

        # # Получаем все сигналы по функции
        # self.signals = self.get_signals(percent=1.8, price_history=prices)
        #
        # self.x = []
        # self.y = []
        #
        # old_train_x = 10
        #
        # self.data = np.array(self.data)
        #
        # open = self.data[:, 1].astype(float)
        # open = self.scaler.fit_transform(open.reshape(-1, 1)).flatten()
        #
        # self.signals = self._signals_normalization(self.signals)
        #
        # new_signals = [{'Type': 'HOLD', 'Start': index, 'Price': price} for index, price in enumerate(prices)]
        #
        # for index_main, signal in enumerate(self.signals):
        #     new_signals[signal['Start']]['Type'] = signal['Type']
        #
        # self.signals = new_signals
        #
        # # Построим график по входным данным обучения
        # self.trade_builder = SignalsChartBuilder(self.token, self.data, self.signals)
        # self.trade_builder.build()
        #
        # # Делаем up sampling, расширение данных
        # # Иначе сеть проглотит слишком неравномерное количество равных данных
        # len_max = len([item for item in self.signals if item['Type'] == 'HOLD'])
        #
        # new_arr_sell = [item for item in self.signals if item['Type'] == 'SELL']
        # new_arr_sell *= len_max // len(new_arr_sell)
        #
        # new_arr_buy = [item for item in self.signals if item['Type'] == 'BUY']
        # new_arr_buy *= len_max // len(new_arr_buy)
        #
        # self.signals.extend(new_arr_buy)
        # self.signals.extend(new_arr_sell)
        #
        # indexes = []
        #
        # for signal in self.signals:
        #     if signal['Start'] <= old_train_x:
        #         continue
        #
        #     if signal['Type'] == 'BUY':
        #         y_binary = [0, 0, 1]
        #     elif signal['Type'] == 'SELL':
        #         y_binary = [0, 1, 0]
        #     else:
        #         y_binary = [1, 0, 0]
        #
        #     # Входными данными для X являются данные по свечам
        #     # с момента сигнала и все 5 предыдущих свечей
        #     self.x.append(open[signal['Start'] - old_train_x:signal['Start']])
        #     self.y.append(y_binary)
        #     indexes.append(signal['Start'])
        #
        # self.x = np.array(self.x)
        # self.y = np.array(self.y)
        #
        # # TODO: Тестовые данные просачиваются в тренировочные
        # indexes_train, \
        # indexes_test, \
        # x_train, \
        # x_test, \
        # y_train, \
        # y_test = train_test_split(
        #     indexes,
        #     self.x,
        #     self.y,
        #     test_size=0.20,
        #     shuffle=True
        # )
        # print('indexes_test', indexes_test)
        # return x_train, x_test[old_train_x + 2:], y_train, y_test[old_train_x + 2:], prices, indexes_test[old_train_x + 2:]
