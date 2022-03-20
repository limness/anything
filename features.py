import pandas as pd
from charts import FeaturesChartBuilder


class FeaturesBuilder:
    """Класс для добавления фич внутри данных"""

    def __init__(self, data, features=None, show_features=None, serializer="pkl") -> None:
        if features is None:
            features = ["open_as_is", "high_as_is", "low_as_is", "close_as_is"]
        self.data = data
        self.serializer = serializer
        self.features = features

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

        return df