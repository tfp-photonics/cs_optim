## custom loss function
def MSEloss_fn(y, yhat):
    loss = ((y - yhat) ** 2).mean()
    return loss


def MAEloss_fn(y, yhat):
    loss = abs(y - yhat).mean()
    return loss


def MREloss_fn_norm(y, yhat):
    return (abs(y - yhat) / (abs(y) + 1)).mean()


def MREloss_fn(y, yhat):
    return (abs(y - yhat) / abs(y)).mean()
