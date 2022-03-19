import pickle

from models.model_in_out import ModelInOut
from models.model_in_out import ModelDataBuilder


def start():
    # with open(f'tokens/ADA-USDT.pickle', 'rb') as handle:
    #     klines_per_day = pickle.load(handle)

    # Индексы покупок
    # Индексы сливов
    # Индекс удержания позиции
    # Индекс не покупки позиции
    # Тип

    # Загрузить веса или использовать ранее ссозданные
    load_weights = False
    token = "ADA-USDT"

    # if not load_weights:
    data_builder = ModelDataBuilder(token)


    model_million = ModelInOut(token, data=data_builder.build())
    model_million.stats()

    model_million.predict()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start()
