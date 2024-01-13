import xgboost, numpy, matplotlib.pyplot as pyplot
def get_model_default(X_train, y_train, X_valid, y_valid):
    params_1 = {
        'booster': 'gbtree',
        'tree_method':'hist',
        'max_depth': 8, 
        'learning_rate': 0.025,
        'objective': 'multi:softprob',
        'num_class': 2,
        'eval_metric':'mlogloss'
        }

    d_train = xgboost.DMatrix(X_train.replace({0:numpy.nan}), y_train)
    d_valid = xgboost.DMatrix(X_valid.replace({0:numpy.nan}), y_valid)

    num_round = 10000
    evallist = [(d_train, 'train'), (d_valid, 'eval')]

    return xgboost.train(params_1, d_train, num_round, evallist, early_stopping_rounds=100)
def show_importance(model):
    xgboost.plot_importance(model, max_num_features=25)
    fig = pyplot.gcf()
    fig.set_size_inches(15, 20)
