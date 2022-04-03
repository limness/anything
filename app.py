from models.model_in_out import ModelInOut
from dataset import DataBuilder
from backtest import Backtest

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime


def start_experiment(experiment_name="") -> None:
    """Метод для запуска первоначального сценария нового эксперимента"""

    time_now = datetime.now().strftime("%m%d%Y%H%M%S")
    append_name = f"_{experiment_name}" if experiment_name.strip() != "" else ""
    experiment_name = f"Experiment_{time_now}{append_name}"
    os.makedirs(f"experiments/{experiment_name}")

    token = "ADA-USDT"
    features = ["open_as_is", "high_as_is", "LF"]

    # Начинаем готовить данные для модели
    data_builder = DataBuilder(
        token,
        features=features,
        show_windows=False,
        show_markup=False,
        markup_frequency=100
    )
    data_builder.add_window(
        name="train",
        size=1200
    )
    data_builder.add_window(
        name="val",
        size=("train", 0.4)
    )
    data_builder.add_window(
        name="test",
        size=500,
        features_by_patch=False
    )
    data_builder.compile_windows()

    # Запускаем обучение сети
    model_million = ModelInOut(
        token,
        data_builder.windows["train"]["Patches"],
        data_builder.windows["val"]["Patches"],
        data_builder.windows["test"]["Patches"],
        show_stats=False,
        experiment_name=experiment_name
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

    backtest = Backtest(signals=data, balance=1000, fix_deal=0.1, commision=0.1)
    backtest.run()
    backtest.show_stats()
    backtest.draw()


def start_test() -> None:
    """Метод для запуска первоначального сценария проверки"""
    token = "ADA-USDT"
    features = ["open_as_is", "high_as_is", "LF"]
    data_builder = DataBuilder(
        token,
        features=features,
        show_windows=True,
        show_markup=False,
        markup_frequency=50
    )
    data_builder.add_window(
        name="test",
        size=1200,
        features_by_patch=True,
    )
    data_builder.compile_windows()
    # print(data_builder.data.shape[0])
    # print(data_builder.windows)
    # model_million = ModelInOut(
    #     token,
    #     test_generator=data_builder.test_generator,
    #     load_model=True
    # )
    # model_million.predict()


if __name__ == '__main__':
    start_experiment(experiment_name="Testchenit")
    # start_test()
