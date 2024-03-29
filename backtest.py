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

    def create_new_order(self, token: str, price: float, commision: float, time: datetime) -> None:
        """Создаем новую позицию на аккаунте"""

        if token in [item['Token'] for item in self._order_book]:
            return

        self._order_book.append({
            'Token': token,
            'Amount': self.fix_amount,
            'Price': price,
            'Time': time
        })
        self._balance -= (self.fix_amount * price) + (self.fix_amount * price) / 100 * commision
        self.min_balance = min(self.min_balance, self._balance)

    def close_all_orders(self, token: str, price: float, commision: float) -> None:
        """Закрываем все позиции определенного токена на аккаунте"""
        total_amount = 0
        for token in self._order_book:
            total_amount += token['Amount']
        self._order_book.clear()
        # index = 0
        # while index < len(self._order_book):
        #     if self._order_book[index]['Token'] == token:
        #         total_amount += self._order_book[index]['Amount']
        #         del self._order_book[index]
        #         continue
        #     index += 1
        self._balance += (total_amount * price) - (total_amount * price) / 100 * commision

    def get_order_book(self) -> []:
        """Возвращаем информацию о текущих открытых
        позициях на аккаунте"""
        return self._order_book

    def get_summary_balance(self, price: float) -> float:
        """Возвращаем общую сводку по балансу"""
        total_amount = 0
        for token in self._order_book:
            total_amount += token['Amount']

        return total_amount * price + self._balance


class VirtualMarket:
    """Класс для синтетической торговли по заранее
    подготовленным данным"""

    def __init__(self, signals: pd.DataFrame, account: VirtualAccount, commision: float = 0.1):
        self._account = account
        self.commision = commision
        self.signals = signals

    def start_virtual_trade(self) -> []:
        """Начинаем симуляцию торговли по заранее
        заданным значениям"""

        # Будем вести логи PNL в течении всей торговли
        stats = {"PNL": []}
        for index, row in self.signals.iterrows():
            ask_price = row["Open"]
            bid_price = row["Open"] + row["Open"] / 100 * 0.001
            # покупаем
            if row['Signal'] == 1:
                # print("sue", index)
                self._account.create_new_order('ADA', bid_price, self.commision, index)
            # продаем
            elif row['Signal'] == -1:
                self._account.close_all_orders('ADA', ask_price, self.commision)
            stats["PNL"].append(self._account.get_summary_balance(ask_price))
        # print("Total balance: {A} Min Balance: {B}".format(
        #     A=self._account.get_summary_balance(ask_price),
        #     B=self._account.min_balance)
        # )
        return stats


class Backtest:
    """Класс для синтетической торговли по заранее
    подготовленным данным"""

    def __init__(self, signals: pd.DataFrame, balance: float, fix_deal: float, commision: float = 0.1):
        self.signals = signals
        self.commision = commision
        self.balance = balance
        self.fix_deal = fix_deal
        self.stats = {}

    def run(self) -> None:
        """Запускаем торговлю"""
        # Создаем аккаунт для торговли
        account = VirtualAccount(balance=self.balance, fix_deal=self.fix_deal)
        # Запускаем игру аккаунта на рынке
        market = VirtualMarket(self.signals, account, self.commision)
        self.stats = market.start_virtual_trade()

    def show_stats(self) -> None:
        """Показываем общую статистику проведения бэктеста"""
        print("+" * 10 + "[ Backtest Anything ]" + "+" * 10)
        print("В разработке...")
        print("+" * 10 + "+++++++++++++++++++++" + "+" * 10)

    def draw(self) -> None:
        """Показываем график торговли"""
        BacktestChartBuilder(self.signals, self.stats).draw()
