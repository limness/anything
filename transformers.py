from typing import Any
from scipy.signal import butter, filtfilt


def as_is(dataset, source, params) -> Any:
    """
    Трансформер для возврата исходной информации

    :param dataset: Исходный датасет
    :param source: Колонка от которой делаем фичу
    :param params: Дополнительные параметры для фичи
    :return:
    """
    df = dataset[source]

    return df


def deriv(dataset, source, params) -> Any:
    """
    Трансформер для формирования производной

    :param dataset: Исходный датасет
    :param source: Колонка от которой делаем фичу
    :param params: Дополнительные параметры для фичи
    :return:
    """
    df = dataset[source].diff()

    return df


def shift(dataset, source, params) -> Any:
    """
    Трансформер для формирования сдвига

    :param dataset: Исходный датасет
    :param source: Колонка от которой делаем фичу
    :param params: Дополнительные параметры для фичи
    :return:
    """
    df = dataset[source].shift()

    return df


def nq(dataset, source, params) -> Any:
    """
    Трансформер для формирования NQ фильтра

    :param dataset: Исходный датасет
    :param source: Колонка от которой делаем фичу
    :param params: Дополнительные параметры для фичи
    :return:
    """

    nq_fs = params.get("fs", 40)
    df = butter_lowpass_filter(dataset[source], fs=nq_fs)

    return df


def butter_lowpass_filter(data, cutoff=2, fs=30.0, order=2):
    """Метод для формирования фильтра в качестве фичи"""
    # Nyquist Frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y