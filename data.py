from crossValidation import Crossvalidation
from prepareData import Data


def getData(set):
    """
    funkcja tworząca obiekt klasy Data, który wczutyje odpowiedni zbiór danych
    :param set: nazwa zbioru danych
    :return:
    """
    if set == 'parkinson':
        # parkinson
        data = Data("parkinson.csv")
        data.toFloat()
        data.Normalize(22)
        cross_data = Crossvalidation(data.features, 10, -1)
    elif set == 'breast-cancer':
        # breast cancer
        data = Data("breast-cancer.csv")
        data.toNumbers()
        data.Normalize(0)
        cross_data = Crossvalidation(data.features, 10, 0)
    elif set == 'heart-disease':
        # heart disease
        data = Data("heart-disease.csv")
        data.toFloat()
        data.Normalize(13)
        cross_data = Crossvalidation(data.features, 10, -1)
    elif set == 'zoo':
        # zoo
        data = Data("zoo.csv")
        data.toNumbers()
        data.Normalize(16)
        cross_data = Crossvalidation(data.features, 10, -1)
    else:
        raise ValueError('Nie ma takiego zbioru danych!')

    return cross_data, data
