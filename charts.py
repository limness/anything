import numpy as np
import matplotlib.pyplot as plt
import trader


class TradeChartBuilder:
    """Класс для построения графиков
    результатов торгов на бирже"""

    def __init__(self, token: str, prices: [], indexes: [], predictions: [], true_dim: []) -> None:
        self.prices = prices
        self.predictions = predictions
        self.indexes = indexes
        self.true_dim = true_dim
        self.token = token

    def build(self) -> None:
        """Метод для вывода статистики трейдинга"""
        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        ax0.grid(color='#2A3459')
        ax1.grid(color='#2A3459')

        self._draw_pred_price_history(ax0)
        self._draw_true_price_history(ax1)

        self._draw_profit_and_loss(ax2)

        plt.legend()
        plt.show()

    def _draw_pred_price_history(self, ax: plt) -> None:
        """A method that draws profit graphic using a strategy"""

        preds = np.array(self.predictions)

        x_buy_signals = [self.indexes[index] for index, pred in enumerate(preds) if np.argmax(pred) == 2]
        x_sell_signals = [self.indexes[index] for index, pred in enumerate(preds) if np.argmax(pred) == 1]
        #
        y_buy_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(preds) if np.argmax(pred) == 2]
        y_sell_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(preds) if np.argmax(pred) == 1]

        ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        ax.plot(self.prices, label="Predicted")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

    def _draw_true_price_history(self, ax: plt) -> None:
        """A method that draws profit graphic using a stratcegy"""

        trues = np.array(self.true_dim)

        x_buy_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
        x_sell_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 1]
        #
        y_buy_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
        y_sell_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 1]

        ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        ax.plot(self.prices, c="grey", label="True")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

    def _draw_profit_and_loss(self, ax: plt) -> None:
        """A method that draws profit graphic using a stratcegy"""

        # Use self.true_dim instead to see what is reference
        data_signals = self.predictions

        # Выполним предварительную обработку
        # TODO: Убрать весь мусор и эту обработку

        signals = [{'Signal': 'HOLD', 'Time': index} for index, price in enumerate(self.prices)]

        for index, signal in enumerate(data_signals):
            if np.argmax(signal) == 2:
                signal_type = 'BUY'
            elif np.argmax(signal) == 1:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            signals[self.indexes[index]]['Signal'] = signal_type

        # Для тестов определим виртуального трейдера для синтетической торговли
        account = trader.VirtualAccount()
        virtual_trader = trader.VirtualTrader(account)

        # Начинаем виртуальную торговлю
        stats = virtual_trader.start_virtual_trade(signals, self.prices)

        #######

        # trues = np.array(self.true_dim)

        # x_buy_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
        # x_sell_signals = [self.indexes[index] for index, pred in enumerate(trues) if np.argmax(pred) == 1]
        # #
        # y_buy_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 2]
        # y_sell_signals = [self.prices[self.indexes[index]] for index, pred in enumerate(trues) if np.argmax(pred) == 1]
        #
        # ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        # ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        ax.plot(stats, label="PNL")
        ax.set_xlabel("Time")
        ax.set_ylabel("PNL")


class SignalsChartBuilder:
    """Класс для построения графиков
    сигналов для выхода в сеть"""

    def __init__(self, token: str, norm_data: [], signals: []) -> None:
        self.norm_data = norm_data
        self.signals = signals
        self.token = token

    def build(self) -> None:
        """Метод для вывода статистики трейдинга"""
        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))

        ax0.grid(color='#2A3459')
        ax1.grid(color='#2A3459')

        self._draw_price_history(self.token, self.norm_data, ax0)
        self._draw_signals_history(self.token, self.norm_data, self.signals, ax1)

        plt.legend()
        plt.show()

    def _draw_price_history(self, token: str, prices: [], ax: plt) -> None:
        """A method that draws profit graphic using a strategy"""

        ax.plot(prices[:, 1].astype(float), label=token)

    def _draw_signals_history(self, token: str, prices: [], signals: [], ax: plt) -> None:
        """A method that draws profit graphic using a strategy"""

        x_buy_signals = [signal['Start'] for signal in signals if signal['Type'] == 'BUY']
        x_sell_signals = [signal['Start'] for signal in signals if signal['Type'] == 'SELL']
        # x_hold_signals = [signal['Start'] for signal in signals if signal['Type'] == 'HOLD']

        y_buy_signals = [signal['Price'] for signal in signals if signal['Type'] == 'BUY']
        y_sell_signals = [signal['Price'] for signal in signals if signal['Type'] == 'SELL']
        # y_hold_signals = [signal['Price'] for signal in signals if signal['Type'] == 'HOLD']

        ax.plot(prices[:, 1].astype(float), label=token)
        ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        # ax.scatter(x_hold_signals, y_hold_signals, label="HOLD", s=10, c='orange')

        ax.legend()


class LearningChartBuilder:
    """Класс для построения графиков
    результатов обучения сети"""

    def __init__(self) -> None:
        self.history = {}

    def set_history(self, history) -> None:
        """Метод для загрузки данных обучения"""
        self.history = history

    def build(self) -> None:
        """Метод для вывода статистики трейдинга"""
        plt.style.use("seaborn-dark")

        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
            plt.rcParams[param] = '0.9'

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 8))

        ax0.grid(color='#2A3459')
        ax1.grid(color='#2A3459')

        ax0.plot(self.history['loss'], label="Loss")
        ax0.plot(self.history['val_loss'], label="Val Loss")

        ax1.plot(self.history['accuracy'], label="Accuracy")
        ax1.plot(self.history['val_accuracy'], label="Val Accuracy")

        plt.legend()
        plt.show()


class ChartBuilder:
    """Класс для построения графиков
    результатов обучения сети"""

    def __init__(self, token, data,
                 train_index, val_index, test_index,
                 train_window=1000, val_window=0.4, test_window=300) -> None:
        self.data = data
        self.token = token

        self.train_index = train_index
        self.val_index = val_index
        self.test_index = test_index

        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window

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

        ax0.plot(self.data.index[self.train_index:self.train_index + self.train_window],
                 self.data["Close"][self.train_index:self.train_index + self.train_window], label="Train")

        ax0.plot(self.data.index[self.val_index:self.val_index + self.val_window],
                 self.data["Close"][self.val_index:self.val_index + self.val_window], label="Val")

        ax0.plot(self.data.index[self.test_index:self.test_index + self.test_window],
                 self.data["Close"][self.test_index:self.test_index + self.test_window], label="Test")

        ax0.set_xlabel("Time")
        ax0.set_ylabel("Price")
        ax0.grid(color='#2A3459')

        plt.legend()
        plt.show()
