import os
from datetime import datetime

import pandas as pd
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import concatenate, SpatialDropout2D, AveragePooling2D

from charts import LearningChartBuilder


class ModelInOut:
    """Класс для построения модели распознавания
    входов выходов и удержания позиции отдельного токена"""

    def __init__(self, token: str, train_generator=None, val_generator=None, test_generator=None,
                 y_scaler=None, load_model=None, save_model=None, show_stats=False, input_layer=(30, 3)) -> None:
        self.token = token
        self.load_model = load_model
        self.save_model = save_model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.show_stats = show_stats
        self.y_scaler = y_scaler
        self.input_layer = input_layer
        self.history = {}

        # Строим архитектуру для модели
        self._build_model()

        # Начинаем обучать модель
        self._train_model()

    def _build_model(self) -> None:
        """Метод для построения архитектуры модели"""
        input = Input(shape=(*self.input_layer, 1))
        k_size = (2, 5)
        dropout = 0.5

        x = Conv2D(32, k_size, padding='same', activation='elu')(input)
        x = SpatialDropout2D(dropout)(x)
        x = Conv2D(32, k_size, padding='same', activation='elu')(x)
        xa = AveragePooling2D(pool_size=(2, 1), padding='same')(x)
        xm = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        xa = Conv2D(32, k_size, padding='same', activation='elu')(xa)
        xm = Conv2D(32, k_size, padding='same', activation='elu')(xm)
        x = concatenate([xa, xm])

        x = Conv2D(128, k_size, padding='same', activation='elu')(x)
        x = SpatialDropout2D(dropout)(x)
        x = Conv2D(128, k_size, padding='same', activation='elu')(x)
        xa = AveragePooling2D(pool_size=(2, 1), padding='same')(x)
        xm = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        xa = Conv2D(64, k_size, padding='same', activation='elu')(xa)
        xm = Conv2D(64, k_size, padding='same', activation='elu')(xm)
        x = concatenate([xa, xm])

        x = Conv2D(128, k_size, padding='same', activation='elu')(x)
        x = SpatialDropout2D(dropout)(x)
        x = Conv2D(128, k_size, padding='same', activation='elu')(x)
        xa = AveragePooling2D(pool_size=(2, 1), padding='same')(x)
        xm = MaxPooling2D(pool_size=(2, 1), padding='same')(x)
        xa = Conv2D(64, k_size, padding='same', activation='elu')(xa)
        xm = Conv2D(64, k_size, padding='same', activation='elu')(xm)
        x = concatenate([xa, xm])

        x = Conv2D(32, k_size, padding='same', activation='elu')(x)
        x = Conv2D(8, k_size, padding='same', activation='elu')(x)
        x = SpatialDropout2D(dropout)(x)
        x = Flatten()(x)
        x = Dense(16, activation='tanh')(x)
        x = Dropout(dropout)(x)
        # x = Dense(2, activation='softmax')(x)

        output = Dense(2, activation='softmax')(x)

        self.model = Model(input, output)
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['binary_accuracy'])

    def _train_model(self) -> None:
        """Метод для запуска обучения или загрузки весов модели"""
        assert self.load_model is not None or self.save_model is not None, \
            "Save/Load directory is empty!"
        if self.load_model is not None:
            self.model.load_weights(f"experiments/{self.load_model}/model.h5")
        else:
            # model_checkpoint_callback = ModelCheckpoint(
            #     filepath='',
            #     save_weights_only=True,
            #     monitor='val_accuracy',
            #     mode='max',
            #     save_best_only=True)
            self.history = self.model.fit(
                self.train_generator["X"],
                self.train_generator["Y"],
                batch_size=128,
                epochs=150,
                validation_data=(self.val_generator["X"], self.val_generator["Y"]),
               # callbacks=[model_checkpoint_callback],
                verbose=1,
                shuffle=True
            )
            self.model.save_weights(f"experiments/{self.save_model}/model.h5")
            self.stats()

    def _evaluate_model(self) -> None:
        """Метод для проверки модели на тестовых данных"""
        stats = self.model.evaluate(self.test_generator["X"], self.test_generator["Y"])
        print('Evaluate:', stats)

    def predict(self) -> pd.DataFrame:
        """Метод для прогноза цены"""
        assert self.test_generator is not None, \
            "Test generator is empty!"
        predictions = self.model.predict(self.test_generator["X"])
        print(predictions)
        predictions = self.y_scaler.inverse_transform(predictions).flatten()

        signals = pd.DataFrame(predictions, columns=["Signal"], index=self.test_generator["Data"].index)
        signals = pd.concat([self.test_generator["Data"], signals], axis=1)
        signals["Signal"] = signals["Signal"].shift()
        signals = signals.dropna()

        for index, pred in enumerate(predictions):
            print("pred", pred, self.test_generator["Y"][index])
        # for true in self.test_generator["Y"]:
        #     print("true", true)

        return signals

    def stats(self) -> None:
        """Метод для вывода статистики тренировки модели"""
        self._evaluate_model()

        if self.show_stats:
            LearningChartBuilder(self.history.history, save_model=self.save_model).draw()
