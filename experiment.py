

class ExperimentBuilder:
    """Класс для подготовки эксперимента"""

    def __init__(self, token: str) -> None:
        self.token = token
        self.patch_size = patch_size
        self.features = features
        self.serializer = serializer
        self.save_serializer = save_serializer
        self.markup_frequency = markup_frequency
        self.show_markup = show_markup
        self.show_features = show_features

    def _try_load_or_make_dataset(self) -> tuple:
        """Метод для формирования первоначального датасета"""
        try:
            # Попробуем загрузить файл из директории
            data = pd.read_csv(f"datasets/{self.token}_.csv")
        except FileNotFoundError:
            # Файл не найден, необходимо сгенерировать датасет с нуля
            # для этого выгрузим не отформатированные данные из каталога _no_format
            data = self._read_dataset_from_file()[:5000, :-1]

        return data