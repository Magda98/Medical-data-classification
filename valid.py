

def valid_classification(out, y):
    out = out.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    x = abs(out - y)
    valid = sum(i < 0.5 for i in x[0])
    percent = valid / x.shape[1] * 100
    return percent