

class VirtualAccount:
    """Класс для хранения данных аккаунта торговца"""

    def __init__(self):
        # Денежный баланс в USDT
        self._balance = 1000
        self.min_balance = 99999

        # Число открытых позиций
        self._order_book = []

    def set_balance(self, balance) -> None:
        """Обновляем информацию о балансе"""
        self._balance = balance

    def get_balance(self) -> float:
        """Возвращаем информацию о балансе"""
        return self._balance

    def create_new_order(self, token: str, amount: float, price: float, time: int) -> None:
        """Создаем новую позицию на аккаунте"""

        if token in [item['Token'] for item in self._order_book]:
            return

        self._order_book.append({
            'Token': token,
            'Amount': amount,
            'Price': price,
            'Time': time
        })
        self._balance -= (amount * price) + (amount * price) / 100 * 0.1
        self.min_balance = min(self.min_balance, self._balance)

    def close_all_orders(self, token: str, price: float) -> None:
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
        self._balance += (total_amount * price) - (total_amount * price) / 100 * 0.1

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


class VirtualTrader:
    """Класс для синтетической торговли по заранее
    подготовленным данным"""

    def __init__(self, account: VirtualAccount):
        self._account = account

    def start_virtual_trade(self, signals: [], prices: []) -> []:
        """Начинаем симуляцию торговли по заранее
        заданным значениям"""

        # Будем вести логи PNL в течении всей торговли
        stats = []
        for index, price in enumerate(prices):
            ask_price = price
            bid_price = price + price / 100 * 0.1

            # print(type(signals[index]['Signal']), signals[index]['Signal'], len(signals), index, len(prices))
            if signals[index]['Signal'] == 'BUY':
                # покупаем
                self._account.create_new_order('ADA', 9.0, bid_price, index)
            elif signals[index]['Signal'] == 'SELL':
                # продаем
                self._account.close_all_orders('ADA', ask_price)
            stats.append(self._account.get_summary_balance(price))

        print("Total balance: {A} Min Balance: {B}" . format(
            A=self._account.get_summary_balance(price),
            B=self._account.min_balance)
        )

        return stats
