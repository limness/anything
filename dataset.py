import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from typing import Union
from sklearn import preprocessing
from charts import WindowsChartBuilder
from features import FeaturesBuilder
from markups import MarkupBuilder


class DataBuilder:
    """
    Класс для подготовки данных для модели

    @ patch_size - Размер выборки по которым модель будет делать прогноз из прошлого (патча)
    @ features - Список используемых фич для обучения модели
    @ save_serializer - Сохранять полученный датасет без/с фич(ами) в конце
    @ show_features - Показывать текущие фичи в датасете
    @ show_forward - Показывать полученные окна на графике
    @ embed_train - Сделать вложенное проверочное окно
    @ step_window - Отступ окон с начала датасета
    @ from_end - Перенести окна в конец датасета
    @ train_window - Размер тренировочного окна
    @ val_window - Размер валидационного окна
    @ test_window - Размер тестового окна
    @ train_index - Индекс начала тренировочного окна
    @ val_index - Индекс начала валидационного окна
    @ test_index - Индекс начала тестового окна

    """

    def __init__(self, token: str, train_window=None, val_window=0.4, test_window=None,
                 features=None, save_serializer=False, serializer="csv",
                 patch_size=30, markup_frequency=10.0, show_features=None, show_windows=False, show_markup=False,
                 embed_train=False, from_end=True, step_window=500) -> None:
        self.token = token
        self.patch_size = patch_size
        self.features = features
        self.serializer = serializer
        self.save_serializer = save_serializer
        self.markup_frequency = markup_frequency
        self.show_markup = show_markup
        self.show_features = show_features
        self.show_windows = show_windows
        self.embed_train = embed_train
        self.step_window = step_window
        self.from_end = from_end

        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.windows = {}
        self.x_scaler, self.y_scaler = self._try_load_or_make_scaler()
        self.data = self._try_load_or_make_dataset()
        self.targets = self._try_load_or_make_markup(self.data)

    def _try_load_or_make_dataset(self) -> ():
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
            data['Datetime'] = pd.to_datetime(data['Datetime'].astype('int64'), unit='s')
            data = data.set_index("Datetime")

            if self.save_serializer:
                # Сохраним результаты в файл, что в дальнейшем выгружать все из кеша
                data.to_csv(f"datasets/{self.token}_.csv")

        return data

    def _try_load_or_make_markup(self, data) -> ():
        """Метод для формирования разметки датасета"""
        try:
            # Попробуем загрузить файл из директории
            targets = pd.read_csv(f"datasets/{self.token}_targets_.csv")
        except FileNotFoundError:
            # Делаем разметку по всему датасету
            targets = MarkupBuilder(data, self.markup_frequency, self.show_markup).make_markup()
            print(targets)
            if self.save_serializer:
                # Сохраним результаты в файл, что в дальнейшем выгружать все из кеша
                targets.to_csv(f"datasets/{self.token}_targets_.csv")

        return targets

    def _try_load_or_make_scaler(self) -> tuple:
        """Метод для формирования скейлера датасета"""
        # TODO: Сделать загрузку скейлера
        x_scaler = preprocessing.MinMaxScaler()
        y_scaler = preprocessing.OneHotEncoder()
        return x_scaler, y_scaler

    def add_window(self, name: str, size: Union[int, tuple], features_by_patch: bool = False):
        """Метод для инициализации нового окна"""
        # Если в качестве размера передан кортеж
        if isinstance(size, tuple):
            assert size[0] in self.windows, \
                f"{size[0]} not found in dataset dictionary! There is/are {self.windows.keys()}"
            # Извлекаем окно, которое было указано в размере
            # и получаем новый размер умножая на коэффициент
            size = int(self.windows[size[0]]["Size"] * size[1])
        # Конвертнем словарь в список
        windows = list(self.windows.keys()) # [-1]
        if len(windows) > 0:
            # Получаем последнее добавленное окно
            last_window = self.windows[windows[-1]]
            # Получаем позицию последнего окна и устанавливаем старт от него
            start_index = last_window["Start"] + last_window["Size"]
        else:
            # Это первое окно, устанавливаем позицию как нулевую
            start_index = 0
        # Если необходимо брать окно с конца, переносим все
        self.windows[name] = {"Start": start_index, "Size": size, "Features": features_by_patch}

    def compile_windows(self):
        """Метод для компиляции всех окон"""
        # Проверяем включено ли использование окон с конца
        if self.from_end:
            # Проходимся по всем окнам с конца и переносит их в конец датасета
            keys = list(self.windows.keys())
            preview_index = self.data.shape[0]
            for key in reversed(keys):
                self.windows[key]["Start"] = preview_index - self.windows[key]["Size"]
                preview_index = self.windows[key]["Start"]
        # Проходимся по всем окнам для масштабирования
        for key, item in self.windows.items():
            start_index = self.windows[key]["Start"]
            end_index = self.windows[key]["Start"] + self.windows[key]["Size"]
            features_by_patch = self.windows[key]["Features"]
            # Формируем данные окна вместе с фичами
            data, featurized_data, targets = self.__form_window(start_index, end_index, features_by_patch)
            # Формируем данные патчей внутри окна
            patches = self.__form_patches(featurized_data, targets)
            # Создаем новый ключ и записываем данные патча
            self.windows[key]["Patches"] = patches
            # Создаем новый ключ и записываем данные обучения
            self.windows[key]["Data"] = data[self.patch_size:]
        # Если включено отображение окон
        if self.show_windows:
            self._show_windows()

    def _show_windows(self):
        """Метод для отображения окон"""
        WindowsChartBuilder(self.token, self.data, self.windows).draw()

    def __form_window(self, start_index: int, end_index: int, features_by_patch: bool) -> tuple:
        """Метод для обрезания общих данных до нужных окон"""
        data = self.data[start_index:end_index]
        targets = self.targets[start_index:end_index]
        targets = self.__scaler_y(targets)
        # Формируем новый датасет, расставляем фичи
        featurized_data = FeaturesBuilder(
            data,
            self.features,
            by_patch=features_by_patch,
            patch_size=self.patch_size
        ).make_features()
        featurized_data = self.__scaler_x(featurized_data)
        return data, featurized_data, targets

    def __form_patches(self, data, targets) -> tf.keras.preprocessing.sequence.TimeseriesGenerator:
        """Метод для формирования патчей из датасета"""
        # Формируем патчи из полученного финального датасета
        patches = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data=data, targets=targets, length=self.patch_size,
            sampling_rate=1, batch_size=64
        )
        print("patch and data", len(patches), len(patches[0]), len(patches[0][0]),
              len(patches[0][0][0]), len(patches[0][0][0][0]), len(patches[0][0][0][0]), self.patch_size, data.shape)
        return patches

    def __scaler_x(self, data: pd.DataFrame) -> np.array:
        """Метод для скалирования входных данных"""
        data = self.x_scaler.fit_transform(data)
        return data

    def __scaler_y(self, data: pd.DataFrame) -> np.array:
        """Метод для скалирования входных данных"""
        # print("aaaa", data)
        data = self.y_scaler.fit_transform(data).toarray()
        # print("bbbb", data)
        return data

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
