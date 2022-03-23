import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
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

    def __init__(self, token: str, features=None, save_serializer=False, serializer="csv",
                 patch_size=30, show_features=None, show_forward=False,
                 embed_train=False, from_end=True, step_window=500,
                 train_window=1000, val_window=0.4, test_window=300) -> None:
        self.token = token
        self.patch_size = patch_size
        self.features = features
        self.serializer = serializer
        self.save_serializer = save_serializer
        self.show_features = show_features
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

        self.data, self.targets = self._try_load_or_make_dataset() #self.featurized_data,
        self.scaler = preprocessing.MinMaxScaler()
        self.train_generator = self._train_window()
        self.val_generator = self._val_window()
        self.test_generator = self._test_window()
        # print(type(self.train_generator))
        # print("=====")
        # print(len(self.train_generator))

    def _try_load_or_make_dataset(self) -> ():
        """Метод для формирования первоначального датасета"""
        try:
            # Попробуем загрузить файл из директории
            data = pd.read_csv(f"datasets/{self.token}_.csv")
            # Попробуем загрузить файл из директории
            featurized_data = pd.read_csv(f"datasets/{self.token}_featurized_.csv")
            # Попробуем загрузить файл из директории
            targets = pd.read_csv(f"datasets/{self.token}_targets_.csv")
        except FileNotFoundError:
            # Файл не найден, необходимо сгенерировать датасет с нуля
            # для этого выгрузим не отформатированные данные из каталога _no_format
            data = self._read_dataset_from_file()[:5000, :-1]
            # Переводим numpy массив в DataFrame, чтобы в дальнейшем было удобно работать с данными
            data = pd.DataFrame(data, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
            data['Datetime'] = pd.to_datetime(data['Datetime'].astype('int64'), unit='s')
            data = data.set_index("Datetime")
            # Делаем разметку по всему датасету
            targets = MarkupBuilder(data).make_markup()
            # # Добавим фичи и формируем новый датасет
            # featurized_data = FeaturesBuilder(data, self.features, self.show_features).make_features()

            if self.save_serializer:
                # Сохраним результаты в файл, что в дальнейшем выгружать все из кеша
                data.to_csv(f"datasets/{self.token}_.csv")
                # Сохраним результаты в файл, что в дальнейшем выгружать все из кеша
                # featurized_data.to_csv(f"datasets/{self.token}_featurized_.csv")
                # Сохраним результаты в файл, что в дальнейшем выгружать все из кеша
                targets.to_csv(f"datasets/{self.token}_targets_.csv")

        print("Dataset", data)
        # print("Featurized Dataset", featurized_data)
        # Сформировать окна для модели
        self._form_index_windows(data)
        # Если включено отображение форвардного анализа - выводим график
        if self.show_forward:
            print("test_index", self.test_index,
                  "train_index", self.train_index,
                  "val_index", self.val_index)
            WindowsChartBuilder(self.token, data,
                         self.train_index, self.val_index, self.test_index,
                         self.train_window, int(self.train_window * self.val_window), self.test_window).draw()
        return data, targets# featurized_data, targets

    def _form_index_windows(self, data: pd.DataFrame) -> None:
        """Метод для формирования индексов окон"""

        # Тренировочная выборка будет определена как начало датасета + шаг окна
        self.train_index = self.step_window
        # Проверочная выборка будет определена как конец тренировочного окна,
        # если выборка будет не вложенная. Иначе умещаем выборку в тренировочную
        self.val_index = self.train_index + self.train_window \
            if not self.embed_train else self.train_index + self.train_window - int(self.train_window * self.val_window)
        # Тестовая выборка будет определена как конец валидационного окна
        self.test_index = self.val_index + int(self.val_window * self.train_window)

        # Если необходимо брать окна с конца, переносим все
        if self.from_end:
            # Отнимем от общего размера данных конец нашего последнего окна
            # и получим общий индекс свдига
            index_offset = data.shape[0] - (self.test_index + self.test_window)

            self.train_index += index_offset
            self.val_index += index_offset
            self.test_index += index_offset

    def __form_patches(self, data, targets) -> object:
        """Метод для формирования патчей из датасета"""
        patches = tf.keras.preprocessing.sequence.TimeseriesGenerator(
            data=data.to_numpy(), targets=targets.to_numpy(), length=self.patch_size,
            sampling_rate=1, batch_size=64
        )
        return patches

    def patches_generator(former) -> object:
        """Метод для формирования данных для обучения модели"""
        def wrapper(self):
            data, targets = former(self)
            # Добавим фичи, формируем новый датасет
            featurized_data = FeaturesBuilder(data, self.features, by_patch=True).make_features()
            # Формируем патчи из полученного финального датасета
            patches = self.__form_patches(featurized_data, targets)
            return patches
        return wrapper

    @patches_generator
    def _train_window(self) -> tuple:
        """Метод для обрезания общих данных до нужных окон"""
        data = self.data[self.train_index:self.train_index + self.train_window]
        targets = self.targets[self.train_index:self.train_index + self.train_window]
        return data, targets

    @patches_generator
    def _val_window(self) -> tuple:
        """Метод для обрезания общих данных до нужных окон"""
        data = self.data[self.val_index:self.val_index + self.val_window]
        targets = self.targets[self.val_index:self.val_index + self.val_window]
        return data, targets

    @patches_generator
    def _test_window(self) -> tuple:
        """Метод для обрезания общих данных до нужных окон"""
        data = self.data[self.test_index:self.test_index + self.test_window]
        targets = self.targets[self.test_index:self.test_index + self.test_window]
        return data, targets

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
