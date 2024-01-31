import pandas, xgboost, numpy
from sklearn import model_selection
from .model import Model

class XgboostModel(Model):
  def __init__(self, X_train, y_train, train_scores, \
        params={
          'booster': 'gbtree',
          'tree_method':'hist',
          'max_depth': 8, 
          'learning_rate': 0.025,
          'objective': 'multi:softprob',
          'num_class': 2,
          'eval_metric':'mlogloss'
        }):
    Model.__init__(self, X_train, y_train, train_scores, params)
    self.model = None    
  def fit(self):
    train_size=0.8
    random_state=42
    num_round = 10000
    y_train = self.format_y(self.y_train)
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(self.X_train, y_train, train_size=train_size, random_state=random_state)
    d_train = xgboost.DMatrix(self.format_x(X_train), y_train)
    d_valid = xgboost.DMatrix(self.format_x(X_valid), y_valid)
    evallist = [(d_train, 'train'), (d_valid, 'eval')]
    params = self.params.copy()
    params['num_class'] = 3
    self.model = xgboost.train(params, d_train, num_round, evallist, early_stopping_rounds=100, verbose_eval=0)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict(xgboost.DMatrix(self.format_x(X)), iteration_range=(0, self.model.best_iteration)))
    predictions.columns = [0,1,2]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)