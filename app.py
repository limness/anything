from models.model_in_out import ModelInOut
from dataset import DataBuilder
from backtest import Backtest

import os
from datetime import datetime


def create_experiment_directory(experiment_name: str) -> str:
    """Метод для создания директории нового эксперимента"""
    time_now = datetime.now().strftime("%m%d%Y%H%M%S")
    append_name = f"_{experiment_name}" if experiment_name.strip() != "" else ""
    experiment_name = f"Experiment_{time_now}{append_name}"
    os.makedirs(f"experiments/{experiment_name}")
    return experiment_name


def save_features(experiment_name: str, features: list) -> None:
    """Метод для создания директории нового эксперимента"""
    with open(f"experiments/{experiment_name}/features.txt", "+w") as writer:
        features_str = ", ".join([feature['Name'] for feature in features])
        writer.write(features_str)


def load_features(experiment_name: str) -> list:
    """Метод для выгрузки фич из директории эксперимента"""
    with open(f"experiments/{experiment_name}/features.txt", "+w") as reader:
        features = reader.read().split(", ")
    return features


def start_experiment(token, features, experiment_name="") -> None:
    """Метод для запуска первоначального сценария нового эксперимента"""
    # Создаем новую директорию
    experiment_name = create_experiment_directory(experiment_name)

    # Сохраняем фичи
    save_features(experiment_name, features)

    patch_size = 30
    # Создаем словарь из путей где хранятся скейлеры
    scalers = {
        "x": f"experiments/{experiment_name}/x_scaler",
        "y": f"experiments/{experiment_name}/y_scaler"
    }
    # Начинаем готовить данные для модели
    data_builder = DataBuilder(
        token,
        features=features,
        show_windows=True,
        show_markup=False,
        markup_frequency=40,
        save_scaler=scalers,
        patch_size=patch_size,
        cut_dataset=1600,
    )
    data_builder.add_window(
        name="train",
        size=400
    )
    data_builder.add_window(
        name="val",
        size=("train", 0.4)
    )
    data_builder.add_window(
        name="test",
        size=1000,
        features_by_patch=True,
        # generate_targets=False,
    )
    data_builder.compile_windows()

    # Запускаем обучение сети
    model_million = ModelInOut(
        token,
        data_builder.windows["train"],
        data_builder.windows["val"],
        data_builder.windows["test"],
        y_scaler=data_builder.y_scaler,
        show_stats=True,
        save_model=experiment_name,
        input_layer=(patch_size, len(features))
    )
    # Получаем сигналы
    signals = model_million.predict()

    # Запускаем бэктест
    backtest = Backtest(
        signals=signals,
        balance=1000,
        fix_deal=0.1,
        commision=0.1,
        save_stats=experiment_name
    )
    backtest.run()
    backtest.show_stats()
    backtest.draw()


def start_test(token, experiment_name="") -> None:
    """Метод для запуска сценария проверки модели на разных временных рядах"""

    # Выгружаем фичи
    features = load_features(experiment_name)
    print("features load", features)
    patch_size = 30

    # Начинаем готовить данные для модели
    data_builder = DataBuilder(
        token,
        features=features,
        show_windows=True,
        show_markup=False,
        markup_frequency=50,
        load_scaler=experiment_name,
        patch_size=patch_size,
    )
    data_builder.add_window(
        name="test",
        size=500,
        features_by_patch=False
    )
    data_builder.compile_windows()

    # Запускаем загрузку модели
    model_million = ModelInOut(
        token,
        test_generator=data_builder.windows["test"],
        y_scaler=data_builder.y_scaler,
        load_model=experiment_name,
        input_layer=(patch_size, len(features))
    )
    # Получаем сигналы
    signals = model_million.predict()

    # Запускаем бэктест
    backtest = Backtest(signals=signals, balance=1000, fix_deal=0.1, commision=0.0, play_short=True)
    backtest.run()
    backtest.show_stats()
    backtest.draw()


if __name__ == '__main__':

    token = "ADA-USDT"
    features = [
        {
            "Name": "Open",
            "Source": "Open",
            "Transformer": "deriv",
            "Type": "Global",
            "Params": {}
        },
        {
            "Name": "High",
            "Source": "High",
            "Transformer": "deriv",
            "Type": "Global",
            "Params": {}
        },
        {
            "Name": "Low",
            "Source": "Low",
            "Transformer": "deriv",
            "Type": "Global",
            "Params": {}
        },
        {
            "Name": "Close",
            "Source": "Close",
            "Transformer": "deriv",
            "Type": "Global",
            "Params": {}
        },
        {
            "Name": "Volume",
            "Source": "Close",
            "Transformer": "deriv",
            "Type": "Global",
            "Params": {}
        },
        {
            "Name": "NQ",
            "Source": "Close",
            "Transformer": "nq",
            "Type": "Global",
            "Params": {"fs": 120}
        },
        {
            "Name": "NQ_Deriv",
            "Source": "NQ",
            "Transformer": "deriv",
            "Type": "Global",
            "Params": {}
        },
        # {
        #     "Name": "Volume",
        #     "Source": "Close",
        #     "Transformer": "deriv",
        #     "Type": "Global",
        #     "Params": {}
        # },
        # {
        #     "Name": "Open",
        #     "Source": "Open",
        #     "Transformer": "deriv",
        #     "Type": "Local",
        #     "Params": {}
        # },
        # {
        #     "Name": "ao_indicator",
        #     "Source": "Open",
        #     "Transformer": "deriv",
        #     "Type": "Local",
        #     "Params": {}
        # }
    ]
    experiment_name = "First_Indicator"

    start_experiment(token, features, experiment_name)
    # start_test(token, experiment_name)
