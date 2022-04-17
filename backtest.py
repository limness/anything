import pandas as pd
from datetime import datetime
from charts import BacktestChartBuilder


class VirtualAccount:
    """Класс для хранения данных аккаунта торговца"""

    def __init__(self, balance, fix_deal):
        # Денежный баланс в USDT
        self._balance = balance
        self.fix_amount = fix_deal
        self.min_balance = 99999

        # Число открытых позиций
        self._order_book = []

    def set_balance(self, balance) -> None:
        """Обновляем информацию о балансе"""
        self._balance = balance

    def get_balance(self) -> float:
        """Возвращаем информацию о балансе"""
        return self._balance

    def create_new_order(self, type: str, token: str, price: float, commision: float, time: datetime) -> None:
        """Создаем новую позицию на аккаунте"""
        if token in [item['Token'] for item in self._order_book]:
            return
        self._order_book.append({
            'Type': type,
            'Token': token,
            'Amount': self.fix_amount,
            'Price': price,
            'Time': time
        })
        self._balance -= (self.fix_amount * price) + (self.fix_amount * price) / 100 * commision
        self.min_balance = min(self.min_balance, self._balance)

    def close_all_orders(self, token: str, price: float, commision: float) -> None:
        """Закрываем все позиции определенного токена на аккаунте"""
        total_orders_balance = 0
        for token in self._order_book:
            # total_amount += token['Amount']
            short = 1 if token['Type'] == "Long" else -1
            total_orders_balance += ((price - token['Price']) * short + token['Price']) * token['Amount']
        self._order_book.clear()
        self._balance += total_orders_balance - total_orders_balance / 100 * commision

    def get_order_book(self) -> []:
        """Возвращаем информацию о текущих открытых
        позициях на аккаунте"""
        return self._order_book

    def get_summary_balance(self, price: float) -> float:
        """Возвращаем общую сводку по балансу"""
        total_orders_balance = 0
        for token in self._order_book:
            short = 1 if token['Type'] == "Long" else -1
            total_orders_balance += ((price - token['Price']) * short + token['Price']) * token['Amount']

        return total_orders_balance + self._balance


class VirtualMarket:
    """Класс для синтетической торговли по заранее
    подготовленным данным"""

    def __init__(self, signals: pd.DataFrame, account: VirtualAccount, commision: float = 0.1,
                 play_short: bool = False):
        self._account = account
        self.commision = commision
        self.signals = signals
        self.play_short = play_short

    def start_virtual_trade(self) -> []:
        """Начинаем симуляцию торговли по заранее
        заданным значениям"""

        # Будем вести логи PNL в течении всей торговли
        stats = {"PNL": []}
        preview_signal = None
        for index, row in self.signals.iterrows():
            ask_price = row["Open"]
            bid_price = row["Open"] + row["Open"] / 100 * 0.001
            # Закрываем предыдущие сделки, если меняем тип ордера
            if preview_signal != row['Signal'] and preview_signal is not None:
                self._account.close_all_orders('ADA', ask_price, self.commision)
            # покупаем
            if row['Signal'] == 1:
                self._account.create_new_order("Long", "ADA", bid_price, self.commision, index)
            # продаем
            elif row['Signal'] == -1:
                self._account.create_new_order("Short", "ADA", bid_price, self.commision, index)
            stats["PNL"].append(self._account.get_summary_balance(ask_price))
            preview_signal = row['Signal']
        return stats


class Backtest:
    """Класс для синтетической торговли по заранее
    подготовленным данным"""

    def __init__(self, signals: pd.DataFrame, balance: float, fix_deal: float,
                 play_short: bool = False, commision: float = 0.1, save_stats=None):
        self.signals = signals
        self.commision = commision
        self.balance = balance
        self.fix_deal = fix_deal
        self.save_stats = save_stats
        self.play_short = play_short
        self.stats = {}

    def run(self) -> None:
        """Запускаем торговлю"""
        # Создаем аккаунт для торговли
        account = VirtualAccount(balance=self.balance, fix_deal=self.fix_deal)
        # Запускаем игру аккаунта на рынке
        market = VirtualMarket(self.signals, account, self.commision, self.play_short)
        self.stats = market.start_virtual_trade()

    def show_stats(self) -> None:
        """Показываем общую статистику проведения бэктеста"""
        print("+" * 10 + "[ Backtest Anything ]" + "+" * 10)
        print("В разработке...")
        print("+" * 10 + "+++++++++++++++++++++" + "+" * 10)

    def draw(self) -> None:
        """Показываем график торговли"""
        BacktestChartBuilder(self.signals, self.stats, self.save_stats).draw()
