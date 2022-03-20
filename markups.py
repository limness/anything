import numpy as np
import pandas as pd


class MarkupBuilder:
    """Класс для расставления разметки для обучения"""

    def __init__(self, data, serializer="pkl") -> None:
        self.data = data
        self.serializer = serializer

    def make_markup(self) -> pd.DataFrame:
        """Метод для формирования разметки"""
        df = pd.DataFrame()
        df["Signal"] = self.data["Open"]
        return df

    def t_make_markup(self, percent: float, candles: []) -> []:
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