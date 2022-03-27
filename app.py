from models.model_in_out import ModelInOut
from dataset import DataBuilder


def start() -> None:
    """Метод для запуска первоначального сценария"""

    token = "ADA-USDT"
    features = ["open_as_is", "high_as_is", "LF"]
    data_builder = DataBuilder(token, features=features, show_forward=False, show_markup=False, markup_frequency=100)
    print(data_builder.data)

    model_million = ModelInOut(
        token,
        data_builder.train_generator,
        data_builder.val_generator,
        data_builder.test_generator,
        show_stats=True
    )
    model_million.predict()
    # model_million.stats()
    # model_million.predict()


if __name__ == '__main__':
    start()
