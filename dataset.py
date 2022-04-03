import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from typing import Union
from sklearn import preprocessing
from charts import WindowsChartBuilder
from experiments.Experiment_04032022202403_Markup_Fix.features import FeaturesBuilder
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
        return patches

    def __scaler_x(self, data: pd.DataFrame) -> np.array:
        """Метод для скалирования входных данных"""
        data = self.x_scaler.fit_transform(data)
        return data

    def __scaler_y(self, data: pd.DataFrame) -> np.array:
        """Метод для скалирования входных данных"""
        data = self.y_scaler.fit_transform(data).toarray()
        return data

    def _read_dataset_from_file(self) -> np.array:
        """Метод для чтения данных с биржи из файла"""
        with open(f'tokens/{self.token}.pickle', 'rb') as handle:
            klines_per_day = pickle.load(handle)
        return np.array(klines_per_day).astype(float)
