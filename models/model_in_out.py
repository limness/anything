import os
from datetime import datetime

import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Model
from sklearn.preprocessing import OneHotEncoder
from charts import LearningChartBuilder


class ModelInOut:
    """Класс для построения модели распознавания
    входов выходов и удержания позиции отдельного токена"""

    def __init__(self, token: str, train_generator=None, val_generator=None, test_generator=None,
                 y_scaler=None, experiment_name="", show_stats=False, load_model=False) -> None:
        self.token = token
        self.load_model = load_model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.show_stats = show_stats
        self.experiment_name = experiment_name
        self.y_scaler = y_scaler
        self.history = {}

        # Строим архитектуру для модели
        self._build_model()

        # Начинаем обучать модель
        self._train_model()

    def _build_model(self) -> None:
        """Метод для построения архитектуры модели"""
        input = Input(shape=(30, 3))
        x = Flatten()(input)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(50, activation='sigmoid')(x)

        output = Dense(2, activation='softmax')(x)

        self.model = Model(input, output)
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['binary_accuracy'])

    def _train_model(self) -> None:
        try:
            # TODO: Придумать иное решение
            # Если загрузка модели не установлена, искуственно вызовем except
            if not self.load_model:
                self.model.load_weights("nope")
            self.model.load_weights("my_checkpoint.h5")
        except Exception as ex:
            """Метод для тренировки модели"""
            self.history = self.model.fit(
                self.train_generator["Patches"],
                batch_size=64,
                epochs=600,
                validation_data=self.val_generator["Patches"],
                verbose=1,
                shuffle=True
            )
            print(self.experiment_name)
            self.model.save_weights(f"experiments/{self.experiment_name}/model.h5")
            self.stats()

    def _evaluate_model(self) -> None:
        """Метод для проверки модели на тестовых данных"""
        stats = self.model.evaluate(self.test_generator["Patches"])
        print('Evaluate:', stats)

    def predict(self) -> pd.DataFrame:
        """Метод для прогноза цены"""
        assert self.test_generator is not None, \
            "Test generator is empty!"
        predictions = self.model.predict(self.test_generator["Patches"])
        print(predictions.shape, self.test_generator["Data"].shape)
        predictions = self.y_scaler.inverse_transform(predictions).flatten()
        # for index, prediction in enumerate(predictions):
        #     print("Прогноз: {A}" . format(A=prediction))
        signals = pd.DataFrame(predictions, columns=["Signal"], index=self.test_generator["Data"].index)
        signals = pd.concat([self.test_generator["Data"], signals], axis=1)
        print("signals", signals)
        return signals

    def stats(self) -> None:
        """Метод для вывода статистики тренировки модели"""
        self._evaluate_model()

        if self.show_stats:
            LearningChartBuilder(self.history.history).draw()
