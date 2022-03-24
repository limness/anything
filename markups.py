import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from charts import MarkupsChartBuilder


class MarkupBuilder:
    """Класс для расставления разметки для обучения"""

    def __init__(self, data, frequency=10.0, show_markup=False, serializer="pkl") -> None:
        self.data = data
        self.show_markup = show_markup
        self.frequency = frequency
        self.serializer = serializer

    def make_markup(self) -> pd.DataFrame:
        """Метод для формирования разметки"""
        df = pd.DataFrame()
        # Сглаживаем ряд, чтобы нейронке было проще работать
        df["Signal"] = MarkupBuilder.butter_lowpass_filter(self.data["Open"], fs=self.frequency)
        # Получаем производные, таким образом видим возрастания и падения графика
        df["Signal"] = df["Signal"].diff()
        # Заменяем все отрицательные числа (падения) на -1 и наоборот
        df['Signal'].values[df['Signal'] > 0] = 1
        df['Signal'].values[df['Signal'] <= 0] = -1
        # Установим в качестве индекса у разметки - дату
        # чтобы в дальнейшем можно было удобнее работать
        df = df.set_index(self.data.index)

        if self.show_markup:
            MarkupsChartBuilder(self.data, df).draw()
        return df

    @staticmethod
    def butter_lowpass_filter(data, cutoff=2, fs=10.0, order=2):
        """Метод для формирования фильтра в качестве фичи"""
        # Nyquist Frequency
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y