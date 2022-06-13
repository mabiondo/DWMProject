

def get_dummy_model(strategy, X, y):
    """
    Return a Dummy model fitted for the given data
    :param strategy: the strategy to use (mean or median)
    :param X: X values
    :param y: y true
    :return: The fitted model with the given data and strategy
    """
    from sklearn.dummy import DummyRegressor
    model = DummyRegressor(strategy=strategy)
    return model.fit(X, y)

def print_metrics(y_true, y_pred):
    """
    Print some metrics, useful to compare models
    :param y_true:
    :param y_pred:
    :return:
    """
    from texttable import Texttable
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
    t = Texttable()
    t.add_rows([['Method', 'Value']])
    t.add_row(["mean absolute error", f'{mean_absolute_error(y_true, y_pred)}'])
    t.add_row(["mean squared error", f'{mean_squared_error(y_true, y_pred)}'])
    t.add_row(["r^2", f'{r2_score(y_true, y_pred)}'])
    t.add_row(["explained variance score", f'{explained_variance_score(y_true, y_pred)}'])
    print(t.draw())


def scale(X_train, X_valid, X_test):
    """
    Scale according with X_train values, then apply the scaling to X_valid and X_test
    :param X_train: X_train
    :param X_valid: X_vaild
    :param X_test: X_test
    :return: X_train scaled, X_valid scaled, X_test scaled
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_valid), scaler.transform(X_test)