from models.model_in_out import ModelInOut
from dataset import DataBuilder


def start() -> None:
    """Метод для запуска первоначального сценария"""

    token = "ADA-USDT"
    data_builder = DataBuilder(token, show_forward=True)
    print(data_builder.data)
    # model_million = ModelInOut(token, data=data_builder.build())
    # model_million.stats()
    # model_million.predict()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
