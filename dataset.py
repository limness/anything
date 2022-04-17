import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
from os.path import exists
from typing import Union
from sklearn import preprocessing
from charts import WindowsChartBuilder
from features import FeaturesBuilder
from markups import MarkupBuilder
import joblib

np.set_printoptions(suppress=True)


class TimeseriesGenerator:

    def __init__(self, data, length, targets=None):
        self.data = data
        self.targets = targets
        self.length = length

    def build(self) -> tuple:
        """Метод для формирования патчей"""
        x = []
        y = []
        for index in range(self.data.shape[0]):
            if self.targets is not None:
                # Так как разметка всегда будет брать 1 бар из будущего
                # необходимо срезать с конца 1 элемент
                if index + self.length + 1 > self.data.shape[0]:
                    break
            else:
                if index + self.length > self.data.shape[0]:
                    break
            bar = index + self.length
            # Вытаскаваем из датасета отдельный патч и добавляем к X разметке
            x.append(self.data[bar - self.length:bar])
            # Вытаскаваем из разметки будущий шаг и добавляем к Y разметке
            if self.targets is not None:
                y.append(self.targets[bar])

        return np.array(x), np.array(y)


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

    def __init__(self, token: str, dataset=None, train_window=None, val_window=0.4, test_window=None,
                 features=None, save_serializer=False, serializer="csv", patch_size=30, cut_dataset=None,
                 markup_frequency=10.0, show_features=None, show_windows=False, show_markup=False,
                 embed_train=False, from_end=True, step_window=500, save_scaler=None, load_scaler=None) -> None:
        self.token = token
        self.dataset = dataset
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
        self.save_scaler = save_scaler
        self.load_scaler = load_scaler
        self.cut_dataset = cut_dataset

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

        if self.dataset is not None:
            # Устанавливаем датасет
            data = self._form_dataframe_from_list(np.array(self.dataset).astype(float))
            return data

        if exists(f"datasets/{self.token}_.csv"):
            # Попробуем загрузить файл из директории
            data = pd.read_csv(f"datasets/{self.token}_.csv")
        else:
            # Файл не найден, необходимо сгенерировать датасет с нуля
            # для этого выгрузим не отформатированные данные из каталога _no_format
            if self.cut_dataset is not None:
                data = self._read_dataset_from_file()[:self.cut_dataset, :-1]
            else:
                data = self._read_dataset_from_file()[:, :-1]
            data = self._form_dataframe_from_list(data)

            if self.save_serializer:
                # Сохраним результаты в файл, что в дальнейшем выгружать все из кеша
                data.to_csv(f"datasets/{self.token}_.csv")

        return data

    def _read_dataset_from_file(self) -> np.array:
        """Метод для чтения данных с биржи из файла"""
        with open(f'tokens/{self.token}.pickle', 'rb') as handle:
            klines_per_day = pickle.load(handle)
        return np.array(klines_per_day).astype(float)

    def _form_dataframe_from_list(self, dataset: np.array) -> pd.DataFrame:
        """Метод для формирования датафрейма из листа"""
        # Переводим numpy массив в DataFrame, чтобы в дальнейшем было удобно работать с данными
        data = pd.DataFrame(dataset, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        data['Datetime'] = pd.to_datetime(data['Datetime'].astype('int64'), unit='s')
        data = data.set_index("Datetime")
        return data

    def _try_load_or_make_markup(self, data) -> Union[list, None]:
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
        if self.load_scaler is not None:
            x_scaler = joblib.load(self.load_scaler["x"])
            y_scaler = joblib.load(self.load_scaler["y"])
        else:
            x_scaler = preprocessing.MinMaxScaler()
            y_scaler = preprocessing.OneHotEncoder()
        return x_scaler, y_scaler

    def add_window(self, name: str, size: Union[int, tuple],
                   features_by_patch: bool = False, generate_targets: bool = True):
        """Метод для инициализации нового окна"""
        # Если в качестве размера передан кортеж
        if isinstance(size, tuple):
            assert size[0] in self.windows, \
                f"{size[0]} not found in dataset dictionary! There is/are {self.windows.keys()}"
            # Извлекаем окно, которое было указано в размере
            # и получаем новый размер умножая на коэффициент
            size = int(self.windows[size[0]]["Size"] * size[1])
        # Конвертнем словарь в список
        windows = list(self.windows.keys())
        if len(windows) > 0:
            # Получаем последнее добавленное окно
            last_window = self.windows[windows[-1]]
            # Получаем позицию последнего окна и устанавливаем старт от него
            start_index = last_window["Start"] + last_window["Size"]
            print(name, start_index)
        else:
            # Это первое окно, устанавливаем позицию как нулевую
            start_index = 0
        window_config = {
            "Start": start_index,
            "Size": size,
            "Features": features_by_patch,
            "Targets": generate_targets
        }
        self.windows[name] = window_config

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
            generate_targets = self.windows[key]["Targets"]
            # Формируем данные окна вместе с фичами
            data, featurized_data, targets = self.__form_window(start_index, end_index,
                                                                features_by_patch, generate_targets)
            # Формируем данные патчей внутри окна
            patches = self.__form_patches(featurized_data, targets)
            # Создаем новый ключ и записываем данные X патча
            self.windows[key]["X"] = patches[0]
            # Создаем новый ключ и записываем данные Y патча
            self.windows[key]["Y"] = patches[1]
            # Создаем новый ключ и записываем данные обучения
            self.windows[key]["Data"] = data[self.patch_size:]
        # Если включено отображение окон
        if self.show_windows:
            self._show_windows()

    def _show_windows(self):
        """Метод для отображения окон"""
        for key, window in self.windows.items():
            print(f"Window {key} starts from {window['Data'].index[0]} and ends to {window['Data'].index[-1]}")
        WindowsChartBuilder(self.token, self.data, self.windows).draw()

    def __form_window(self, start_index: int, end_index: int, features_by_patch: bool, generate_targets: bool) -> tuple:
        """Метод для обрезания общих данных до нужных окон"""
        data = self.data[start_index:end_index]
        if generate_targets:
            targets = self.targets[start_index:end_index]
            targets = self.__scaler_y(targets) #np.array(targets.values)#
        else:
            targets = None
        # Формируем новый датасет, расставляем фичи
        featurized_data = FeaturesBuilder(
            data,
            self.features,
            by_patch=features_by_patch,
            patch_size=self.patch_size,
            show_features=self.show_features
        ).make_features()
        featurized_data = self.__scaler_x(featurized_data) #np.array(featurized_data.values)#
        return data, featurized_data, targets

    def __form_patches(self, data: np.array, targets: np.array) -> tuple:
        """Метод для формирования патчей из датасета"""
        # Формируем патчи из полученного финального датасета
        patches = TimeseriesGenerator(data=data, targets=targets, length=self.patch_size).build()
        return patches

    def __scaler_x(self, data: pd.DataFrame) -> np.array:
        """Метод для скалирования входных данных"""
        if self.load_scaler is not None:
            data = self.x_scaler.transform(data)
        else:
            data = self.x_scaler.fit_transform(data)
            joblib.dump(self.x_scaler, self.save_scaler["x"])
        return data

    def __scaler_y(self, data: pd.DataFrame) -> np.array:
        """Метод для скалирования входных данных"""
        if self.load_scaler is not None:
            data = self.y_scaler.transform(data).toarray()
        else:
            data = self.y_scaler.fit_transform(data).toarray()
            joblib.dump(self.y_scaler, self.save_scaler["y"])
        return data
