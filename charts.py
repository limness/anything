import numpy as np
import matplotlib.pyplot as plt


# class TradeChartBuilder:
#     """Класс для построения графиков
#     результатов торгов на бирже"""
#
#     def __init__(self, token: str, prices: [], indexes: [], predictions: [], true_dim: []) -> None:
#         self.prices = prices
#         self.predictions = predictions
#         self.indexes = indexes
#         self.true_dim = true_dim
#         self.token = token
#
#     def build(self) -> None:
#         """Метод для вывода статистики трейдинга"""
#         plt.style.use("seaborn-dark")
#
#         for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
#             plt.rcParams[param] = '#212946'
#         for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
#             plt.rcParams[param] = '0.9'
#
#         fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
#
#         ax0.grid(color='#2A3459')
#         ax1.grid(color='#2A3459')
#
#         self._draw_pred_price_history(ax0)
#         self._draw_true_price_history(ax1)
#
#         self._draw_profit_and_loss(ax2)
#
#         plt.legend()
#         plt.show()
#
#     def _draw_pred_price_history(self, ax: plt) -> None:
#         """A method that draws profit graphic using a strategy"""
#
#         preds = np.array(self.predictions)
#
#         x_buy_signals = [self.indexes[index] for index, pred in enumerate(preds) if np.argmax(pred) == 2]
#         x_sell_signals = [self.indexes[index] for index, pred in enumerate(preds) if np.argmax(pred) == 1]
#         #
#         y_buy_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(preds) if np.argmax(pred) == 2]
#         y_sell_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(preds) if np.argmax(pred) == 1]
#
#         ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
#         ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
#         ax.plot(self.prices, label="Predicted")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Price")
#
#     def _draw_true_price_history(self, ax: plt) -> None:
#         """A method that draws profit graphic using a stratcegy"""
#
#         trues = np.array(self.true_dim)
#
#         x_buy_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
#         x_sell_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 1]
#         #
#         y_buy_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
#         y_sell_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 1]
#
#         ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
#         ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
#         ax.plot(self.prices, c="grey", label="True")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Price")
#
#     def _draw_profit_and_loss(self, ax: plt) -> None:
#         """A method that draws profit graphic using a stratcegy"""
#
#         # Use self.true_dim instead to see what is reference
#         data_signals = self.predictions
#
#         # Выполним предварительную обработку
#         # TODO: Убрать весь мусор и эту обработку
#
#         signals = [{'Signal': 'HOLD', 'Time': index} for index, price in enumerate(self.prices)]
#
#         for index, signal in enumerate(data_signals):
#             if np.argmax(signal) == 2:
#                 signal_type = 'BUY'
#             elif np.argmax(signal) == 1:
#                 signal_type = 'SELL'
#             else:
#                 signal_type = 'HOLD'
#             signals[self.indexes[index]]['Signal'] = signal_type
#
#         # Для тестов определим виртуального трейдера для синтетической торговли
#         account = trader.VirtualAccount()
#         virtual_trader = trader.VirtualTrader(account)
#
#         # Начинаем виртуальную торговлю
#         stats = virtual_trader.start_virtual_trade(signals, self.prices)
#
#         #######
#
#         # trues = np.array(self.true_dim)
#
#         # x_buy_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
#         # x_sell_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 1]
#         # #
#         # y_buy_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
#         # y_sell_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 1]
#         #
#         # ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
#         # ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
#         ax.plot(stats, label="PNL")
#         ax.set_xlabel("Time")
#         ax.set_ylabel("PNL")
#
#
# class SignalsChartBuilder:
#     """Класс для построения графиков
#     сигналов для выхода в сеть"""
#
#     def __init__(self, token: str, norm_data: [], signals: []) -> None:
#         self.norm_data = norm_data
#         self.signals = signals
#         self.token = token
#
#     def build(self) -> None:
#         """Метод для вывода статистики трейдинга"""
#         plt.style.use("seaborn-dark")
#
#         for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
#             plt.rcParams[param] = '#212946'
#         for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
#             plt.rcParams[param] = '0.9'
#
#         fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))
#
#         ax0.grid(color='#2A3459')
#         ax1.grid(color='#2A3459')
#
#         self._draw_price_history(self.token, self.norm_data, ax0)
#         self._draw_signals_history(self.token, self.norm_data, self.signals, ax1)
#
#         plt.legend()
#         plt.show()
#
#     def _draw_price_history(self, token: str, prices: [], ax: plt) -> None:
#         """A method that draws profit graphic using a strategy"""
#
#         ax.plot(prices[:, 1].astype(float), label=token)
#
#     def _draw_signals_history(self, token: str, prices: [], signals: [], ax: plt) -> None:
#         """A method that draws profit graphic using a strategy"""
#
#         x_buy_signals = [signal['Start'] for signal in signals if signal['Type'] == 'BUY']
#         x_sell_signals = [signal['Start'] for signal in signals if signal['Type'] == 'SELL']
#         # x_hold_signals = [signal['Start'] for signal in signals if signal['Type'] == 'HOLD']
#
#         y_buy_signals = [signal['Price'] for signal in signals if signal['Type'] == 'BUY']
#         y_sell_signals = [signal['Price'] for signal in signals if signal['Type'] == 'SELL']
#         # y_hold_signals = [signal['Price'] for signal in signals if signal['Type'] == 'HOLD']
#
#         ax.plot(prices[:, 1].astype(float), label=token)
#         ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
#         ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
#         # ax.scatter(x_hold_signals, y_hold_signals, label="HOLD", s=10, c='orange')
#
#         ax.legend()
#
#
# class LearningChartBuilder:
#     """Класс для построения графиков
#     результатов обучения сети"""
#
#     def __init__(self) -> None:
#         self.history = {}
#
#     def set_history(self, history) -> None:
#         """Метод для загрузки данных обучения"""
#         self.history = history
#
#     def build(self) -> None:
#         """Метод для вывода статистики трейдинга"""
#         plt.style.use("seaborn-dark")
#
#         for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
#             plt.rcParams[param] = '#212946'
#         for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
#             plt.rcParams[param] = '0.9'
#
#         fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))
#
#         ax0.grid(color='#2A3459')
#         ax1.grid(color='#2A3459')
#
#         ax0.plot(self.history['loss'], label="Loss")
#         ax0.plot(self.history['val_loss'], label="Val Loss")
#
#         ax1.plot(self.history['accuracy'], label="Accuracy")
#         ax1.plot(self.history['val_accuracy'], label="Val Accuracy")
#
#         plt.legend()
#         plt.show()


class FeaturesChartBuilder:
    """Класс для построения графиков
        существующих фич"""

    def __init__(self, featurized_data, show_only) -> None:
        self.featurized_data = featurized_data
        self.show_only = show_only

        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

    def draw(self) -> None:
        """Метод для вывода графика всех разрешенных фич"""

        # Если включено отббражение всех фич, берем общий размер датасета
        # иначе получаем размер разрешенных фич
        total_features = self.featurized_data.shape[1] if "__all__" in self.show_only else len(self.show_only)
        fig, axes = plt.subplots(total_features, 1)

        fig.suptitle('Features', fontsize=16)

        for index, ax in enumerate(axes):
            ax.set_title(self.featurized_data.columns[index])
            ax.plot(self.featurized_data.iloc[:, index])
            ax.set_ylabel("Value")
            ax.grid(color='#2A3459')

        plt.legend()
        plt.show()


class WindowsChartBuilder:
    """Класс для построения графиков
    результатов обучения сети"""

    def __init__(self, token, data, windows: dict) -> None:
        self.data = data
        self.token = token
        self.windows = windows

        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

    def draw(self) -> None:
        """Метод для вывода графика датасета и его
        тренировочных/проверочных параметрах"""

        fig, ax0 = plt.subplots(1, 1, figsize=(13, 3))

        ax0.set_title(f"Train Window {self.token}_1min")
        ax0.plot(self.data.index, self.data["Close"], label="Price")

        for key, item in self.windows.items():
            start_index = item["Start"]
            end_index = item["Start"] + item["Size"]
            ax0.plot(self.data.index[start_index:end_index],
                     self.data["Close"][start_index:end_index], label=key)

        ax0.set_xlabel("Time")
        ax0.set_ylabel("Price")
        ax0.grid(color='#2A3459')

        plt.legend()
        plt.show()


class MarkupsChartBuilder:
    """Класс для построения графиков
    разметки"""

    def __init__(self, data, targets) -> None:
        self.common_data = data.copy()
        self.common_data["Signal"] = targets["Signal"]

        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

    def draw(self) -> None:
        """Метод для вывода графика датасета и его
        разметки"""

        fig, ax0 = plt.subplots(1, 1, figsize=(13, 3))

        ax0.set_title(f"Markups")
        ax0.plot(self.common_data["Close"], label="Train")
        ax0.scatter(self.common_data[self.common_data['Signal'] == 1].index,
                    self.common_data[self.common_data['Signal'] == 1].Close, c="green", s=9.0, label="Buy")
        ax0.scatter(self.common_data[self.common_data['Signal'] == -1].index,
                    self.common_data[self.common_data['Signal'] == -1].Close, c="red", s=9.0, label="Sell")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("Price")
        ax0.grid(color='#2A3459')

        plt.legend()
        plt.show()


class LearningChartBuilder:
    """Класс для построения графиков
    обучения нейронной сети"""

    def __init__(self, history) -> None:
        self.history = history

        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

    def draw(self) -> None:
        """Метод для вывода графика"""

        fig, ax0 = plt.subplots(1, 1, figsize=(13, 3))

        ax0.plot(self.history['loss'], label="Loss")
        ax0.plot(self.history['val_loss'], label="Val Loss")
        ax0.plot(self.history['binary_accuracy'], label="Accuracy")
        ax0.plot(self.history['val_binary_accuracy'], label="Val Accuracy")

        plt.legend()
        plt.show()


class BacktestChartBuilder:
    """Класс для построения графиков
    бэктеста"""

    def __init__(self, signals, stats) -> None:
        self.signals = signals
        self.stats = stats

        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

    def draw(self) -> None:
        """Метод для вывода графика"""
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(13, 3), sharex=True)

        # Формируем график прибыли
        ax0.set_title("PNL")
        ax0.plot(self.signals.index, self.stats["PNL"], label="PNL")
        ax0.set_ylabel("Profit")

        # Формируем общий график тренда и сделки на нем
        ax1.set_title("Deals")
        ax1.plot(self.signals.index, self.signals["Open"], label="Real Trend")
        ax1.scatter(self.signals[self.signals['Signal'] == 1].index,
                    self.signals[self.signals['Signal'] == 1].Close, c="green", s=9.0, label="Buy")
        ax1.scatter(self.signals[self.signals['Signal'] == -1].index,
                    self.signals[self.signals['Signal'] == -1].Close, c="red", s=9.0, label="Sell")
        ax1.set_ylabel("Price")

        # Формируем подграфик сигналов
        ax2.set_title("Signals")
        ax2.plot(self.signals.index, self.signals["Signal"], label="Signals")
        ax2.set_ylabel("Signal")

        plt.legend()
        plt.show()