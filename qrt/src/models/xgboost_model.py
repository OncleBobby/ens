import pandas
from .model import Model

import xgboost, numpy
class XgboostModel(Model):
  def __init__(self, X_train, y_train, X_valid, y_valid, train_scores, \
        params={
          'booster': 'gbtree',
          'tree_method':'hist',
          'max_depth': 8, 
          'learning_rate': 0.025,
          'objective': 'multi:softprob',
          'num_class': 2,
          'eval_metric':'mlogloss'
        }):
    Model.__init__(self, X_train, y_train, X_valid, y_valid, train_scores, params)
    self.model = None    
  def train(self):
    d_train = xgboost.DMatrix(self.X_train.replace({0:numpy.nan}), self.y_train)
    d_valid = xgboost.DMatrix(self.X_valid.replace({0:numpy.nan}), self.y_valid)

    num_round = 10000
    evallist = [(d_train, 'train'), (d_valid, 'eval')]
    self.model = xgboost.train(self.params, d_train, num_round, evallist, early_stopping_rounds=100, verbose_eval=0)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict(xgboost.DMatrix(X), iteration_range=(0, self.model.best_iteration)))
    predictions[2] = 0
    predictions.columns = [0,2,1]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)
