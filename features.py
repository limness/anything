import pandas as pd
import numpy as np
import ta
from scipy.signal import butter, filtfilt
from charts import FeaturesChartBuilder


class FeaturesBuilder:
    """Класс для добавления фич внутри данных"""

    def __init__(self, data, features=None, show_features=None, features_args=None,
                 by_patch=False, patch_size=50, serializer="pkl") -> None:
        if features is None:
            features = ("open_as_is", "high_as_is", "low_as_is", "close_as_is")
        if features_args is None:
            features_args = {}
        self.data = data
        self.serializer = serializer
        self.features = features
        self.features_args = features_args
        self.by_patch = by_patch
        self.patch_size = patch_size

        print(show_features)
        # Если включено отображение фич, рисуем график по разрешенным фичам
        if show_features is not None:
            # print("sue")
            FeaturesChartBuilder(self.featurized_data, show_features).draw()

    def make_features(self) -> pd.DataFrame:
        """Метод для формирования фич"""
        # Формируем глобальные фичи
        global_featurized_data = self._make_global_features(self.data)
        if self.by_patch:
            # Добавляем локальные фичи
            local_featurized_data = self._make_features_by_patch(self.data)
        else:
            # Формирование фич по патчам не включено,
            # формируем локальные фичи как глобальные по всему датасету
            local_featurized_data = self._make_local_features(self.data)
        featurized_data = pd.concat([global_featurized_data, local_featurized_data], axis=1)
        # Заменяем наниты на нули
        featurized_data = featurized_data.fillna(0)
        return featurized_data

    def _make_features_by_patch(self, dataset) -> pd.DataFrame:
        """Метод для формирования фич отдельно по каждому патчу
        Необходимо для симуляции расставления фич, как в реальном времени"""
        df_patches_featurized = pd.DataFrame()

        # Проходимся по всему обрезанному датасету (окну)
        for index in range(dataset.shape[0]):
            if index + self.patch_size > dataset.shape[0]:
                break
            bar = index + self.patch_size
            # Разбиваем окно на отдельный патч
            df_patch = dataset.iloc[bar - self.patch_size:bar]
            # Подаем патч в метод для формирования фич
            df_patch_featurized = self._make_local_features(df_patch)
            # Так как мы пытаемся симулировать прогноз в реальной жизни
            # необходимо из всего патча оставить лишь последний бар
            # по котоорому модель затем будет делать предсказание
            df_patches_featurized = pd.concat([df_patches_featurized, df_patch_featurized.iloc[-1:]])
        return df_patches_featurized

    def _make_global_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Метод для формирования глобальных фич для датасета.
        Глобальные фичи - фичи которые формируются по прошлому или
        по текущим значениям, не зависят от будущего"""
        df = pd.DataFrame(index=dataset.index)

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

        # Добавляем параметр объема
        if "volume_as_is" in self.features:
            df["Volume"] = dataset["Volume"]

        # Добавляем параметр производной закрытия цены
        if "open_derivate" in self.features:
            df["OpenDerivate"] = dataset["Open"].diff()

        if "sma" in self.features:
            indicator_bb = ta.momentum.AwesomeOscillatorIndicator(high=dataset["High"], low=dataset["Low"],
                                                                  window1=5, window2=34)
            # Add Bollinger Bands features
            df['SMA'] = indicator_bb.awesome_oscillator()

        # Заменяем наниты на нули
        df = df.fillna(0)

        return df

    def _make_local_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Метод для формирования локальных фич для датасета.
        Локальные фичи - фичи, которые сильно зависят от данных в будущем"""
        df = pd.DataFrame(index=dataset.index)
        # Добавляем параметр фильтра, используя цену закрытия
        if "LF" in self.features:
            lf_window = self.features_args.get("LF_window", 40)
            df["LF" + str(lf_window)] = FeaturesBuilder.butter_lowpass_filter(dataset["Close"], fs=lf_window)
        # Заменяем наниты на нули
        df = df.fillna(0)
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