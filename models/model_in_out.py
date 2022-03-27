
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Model


class ModelInOut:
    """Класс для построения модели распознавания
    входов выходов и удержания позиции отдельного токена"""

    def __init__(self, token: str, train_generator, val_generator, test_generator, show_stats=False) -> None:
        self.token = token
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.show_stats = show_stats

        # # Построим график по результатам обучения модели
        # self.chart_builder = charts.LearningChartBuilder()

        # Строим архитектуру для модели
        self._build_model()

        # Начинаем обучать модель
        self._train_model()

    def _build_model(self) -> None:
        """Метод для построения архитектуры модели"""
        input = Input(shape=(30, 3))
        x = Flatten()(input)
        x = Dense(600, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(500, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(50, activation='sigmoid')(x)

        output = Dense(2, activation='softmax')(x)

        self.model = Model(input, output)
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    def _train_model(self) -> None:
        """Метод для тренировки модели"""
        print(len(self.train_generator))
        print(len(self.train_generator[0]))
        print(len(self.train_generator[0][0]))
        print(len(self.train_generator[0][0][0]))

        print(len(self.test_generator))
        print(len(self.test_generator[0]))
        print(len(self.test_generator[0][0]))
        print(len(self.test_generator[0][0][0]))
        self.history = self.model.fit(
            self.train_generator,
            # batch_size=64,
            epochs=20,
            validation_data=self.val_generator,
            verbose=1,
            shuffle=True
        )
        self.model.save_weights("my_checkpoint")
        self.stats()
        # self.model.load_weights("my_checkpoint")

    def _evaluate_model(self) -> None:
        """Метод для проверки модели на тестовых данных"""
        stats = self.model.evaluate(self.test_generator)
        print('Evaluate:', stats)

    def predict(self) -> None:
        """Метод для прогноза цены"""
        predictions = self.model.predict(self.test_generator)

        for index, prediction in enumerate(predictions):
            print("Прогноз: {A}" . format(A=prediction))

        # signs = []
        # for index, signal in enumerate(predictions):
        #     signs.append(np.argmax(signal))
        # signs = np.array(signs)
        #
        # t_signs = []
        # for index, signal in enumerate(true_dim):
        #     t_signs.append(np.argmax(signal))
        # t_signs = np.array(t_signs)

        # print("Total preds",
        #       predictions.shape,
        #       "Total hold",
        #       len(signs[signs == 0]),
        #       "Total sell",
        #       len(signs[signs == 1]),
        #       "Total buy",
        #       len(signs[signs == 2]))
        # print("true:")
        # print("Total preds",
        #       true_dim.shape,
        #       "Total hold",
        #       len(t_signs[t_signs == 0]),
        #       "Total sell",
        #       len(t_signs[t_signs == 1]),
        #       "Total buy",
        #       len(t_signs[t_signs == 2]))
        # print(len(self.prices))
        # print(234324, self.prices[29798:])

        # self.trade_builder = TradeChartBuilder(self.token, self.prices, self.indexes, predictions, true_dim)
        # self.trade_builder.build()

        # plt.style.use("seaborn-dark")
        #
        # for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
        #     plt.rcParams[param] = '#212946'
        # for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
        #     plt.rcParams[param] = '0.9'
        #
        # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        #
        # ax.grid(color='#2A3459')
        #
        # x_buy_signals = [signal['Start'] for signal in signals if signal['Type'] == 'BUY']
        # x_sell_signals = [signal['Start'] for signal in signals if signal['Type'] == 'SELL']
        #
        # ax.plot(self.prices[29798:, 1].astype(float), label='BTC USDT')
        # ax.scatter(x_buy_signals, y_buy_signals, label="BUY", s=10, c='green')
        # ax.scatter(x_sell_signals, y_sell_signals, label="SELL", s=10, c='red')
        #
        # plt.legend()
        # plt.show()

    def stats(self) -> None:
        """Метод для вывода статистики тренировки модели"""
        self._evaluate_model()

        if self.show_stats:
            self.chart_builder.set_history(self.history.history)
            self.chart_builder.build()

# data = yf.download("BTC-USD", interval="5m", start="2020-01-01", end="2020-03-01")
# print(data)
