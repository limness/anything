import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from charts import FeaturesChartBuilder


class FeaturesBuilder:
    """Класс для добавления фич внутри данных"""

    def __init__(self, data, features=None, show_features=None, features_args=None,
                 by_patch=False, patch_size=50, serializer="pkl") -> None:
        if features is None:
            features = ["open_as_is", "high_as_is", "low_as_is", "close_as_is"]
        if features_args is None:
            features_args = {}
        self.data = data
        self.serializer = serializer
        self.features = features
        self.features_args = features_args
        self.by_patch = by_patch
        self.patch_size = patch_size

        # Генерируем фичи и получаем новый датасет
        self.featurized_data = self.make_features()

        # Если включено отображение фич, рисуем график по разрешенным фичам
        if show_features is not None:
            FeaturesChartBuilder(self.featurized_data, show_features).draw()

    def make_features(self) -> pd.DataFrame:
        """Метод для формирования фич"""
        if self.by_patch:
            # Сделать фичи отдельно для каждого патча
            return self._make_features_by_patch(self.data)
        else:
            # Сделать фичи сразу по всему датасету
            sss = self._make_features_by_full_dataset(self.data)
            print("featurized butno w", sss.shape)
            return sss

    def _make_features_by_patch(self, dataset) -> pd.DataFrame:
        """Метод для формирования фич отдельно по каждому патчу
        Необходимо для симуляции расставления фич, как в реальном времени"""
        df_patches_featurized = pd.DataFrame()

        # Проходимся по всему обрезанному датасету (окну)
        for index in range(dataset.shape[0]):
            if index + self.patch_size == dataset.shape[0] + 0:
                break
            # Разбиваем окно на отдельный патч
            df_patch = dataset.iloc[index:index + self.patch_size]
            # Подаем патч в метод для формирования фич
            df_patch_featurized = self._make_features_by_full_dataset(df_patch)

            # Так как мы пытаемся симулировать прогноз в реальной жизни
            # необходимо из всего патча оставить лишь последний бар
            # по котоорому модель затем будет делать предсказание
            df_patches_featurized = pd.concat([df_patches_featurized, df_patch_featurized.iloc[-1:]])

        print("featurized", df_patches_featurized.shape)
        return dataset

    def _make_features_by_full_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Метод для формирования фич по всему датасету"""
        df = pd.DataFrame()

        # Добавляем параметр открытия цены
        if "open_as_is" in self.features:
            df["Open"] = dataset["Open"]

        # Добавляем параметр пика цены
        if "high_as_is" in self.features:
            df["High"] = dataset["High"]

        # Добавляем параметр дна цены
        if "low_as_is" in self.features:
            df["Low"] = dataset["Low"]

        # Добавляем параметр закрытия цены
        if "close_as_is" in self.features:
            df["Close"] = dataset["Close"]

        # Добавляем параметр фильтра, используя цену закрытия
        if "LF" in self.features:
            lf_window = self.features_args.get("LF_window", 30)
            df["LF" + str(lf_window)] = FeaturesBuilder.butter_lowpass_filter(dataset["Close"], fs=lf_window)

        return df

    @staticmethod
    def butter_lowpass_filter(data, cutoff=2, fs=30.0, order=2):
        """Метод для формирования фильтра в качестве фичи"""
        # Nyquist Frequency
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y