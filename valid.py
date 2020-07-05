
def valid_classification(out, d):
    """
    funkcja sprawdzająca poprawność klasyfikacji dla danych
    :param out: wyjście sieci
    :param d: wartość oczekiwana
    :return: poprawność klasyfikacji wyrażona w %
    """
    out = out.cpu().detach().numpy()
    d = d.cpu().detach().numpy()
    x = abs(d - out)
    valid = sum(i < 0.5 for i in x[0])
    percent = valid / x.shape[1] * 100
    return percent