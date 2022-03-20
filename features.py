import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from charts import FeaturesChartBuilder


class FeaturesBuilder:
    """Класс для добавления фич внутри данных"""

    def __init__(self, data, features=None, show_features=None, features_args=None, serializer="pkl") -> None:
        if features is None:
            features = ["open_as_is", "high_as_is", "low_as_is", "close_as_is"]
        if features_args is None:
            features_args = {}
        self.data = data
        self.serializer = serializer
        self.features = features
        self.features_args = features_args

        # Генерируем фичи и получаем новый датасет
        self.featurized_data = self.make_features()

        # Если включено отображение фич, рисуем график по разрешенным фичам
        if show_features is not None:
            FeaturesChartBuilder(self.featurized_data, show_features).draw()

    def make_features(self) -> pd.DataFrame:
        df = pd.DataFrame()

        # Добавляем параметр открытия цены
        if "open_as_is" in self.features:
            df["Open"] = self.data["Open"]

        # Добавляем параметр пика цены
        if "high_as_is" in self.features:
            df["High"] = self.data["High"]

        # Добавляем параметр дна цены
        if "low_as_is" in self.features:
            df["Low"] = self.data["Low"]

        # Добавляем параметр закрытия цены
        if "close_as_is" in self.features:
            df["Close"] = self.data["Close"]

        # Добавляем параметр фильтра, используя цену закрытия
        if "LF" in self.features:
            lf_window = self.features_args.get("LF_window", 30)
            df["LF" + str(lf_window)] = FeaturesBuilder.butter_lowpass_filter(self.data["Close"], fs=lf_window)

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