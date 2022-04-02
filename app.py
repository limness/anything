from models.model_in_out import ModelInOut
from dataset import DataBuilder
from backtest import Backtest

import pandas as pd
import numpy as np
import pickle


def start_experiment() -> None:
    """Метод для запуска первоначального сценария нового эксперимента"""

    token = "ADA-USDT"
    features = ["open_as_is", "high_as_is", "LF"]
    data_builder = DataBuilder(token, features=features, show_forward=False, show_markup=False, markup_frequency=100)
    print(data_builder.data)
    #
    model_million = ModelInOut(
        token,
        data_builder.train_generator,
        data_builder.val_generator,
        data_builder.test_generator,
        show_stats=True
    )
    model_million.predict()

    # # TODO: Пофиксить весь этот хлам
    # with open(f'tokens/ADA-USDT.pickle', 'rb') as handle:
    #     klines_per_day = pickle.load(handle)
    #
    # data = pd.DataFrame(np.array(klines_per_day).astype(float)[:5000, :-1],
    #                     columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
    # data['Datetime'] = pd.to_datetime(data['Datetime'].astype('int64'), unit='s')
    # data = data.set_index("Datetime")
    # data["Signal"] = np.random.choice([-1, 1], data.shape[0])
    #
    # print(data)
    #
    # backtest = Backtest(signals=data, balance=1000, fix_deal=0.1, commision=0.1)
    # backtest.run()
    # backtest.stats()
    # backtest.draw()


def start_test() -> None:
    """Метод для запуска первоначального сценария проверки"""
    token = "ADA-USDT"
    features = ["open_as_is", "high_as_is", "LF"]
    data_builder = DataBuilder(
        token,
        features=features,
        show_forward=False,
        show_markup=False,
        markup_frequency=100
    )
    model_million = ModelInOut(
        token,
        test_generator=data_builder.test_generator,
        load_model=True
    )
    model_million.predict()


if __name__ == '__main__':
    # start_experiment()
    start_test()
