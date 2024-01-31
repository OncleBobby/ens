import pandas, xgboost, numpy
from .model import Model
from catboost import CatBoostClassifier

class CatBoostModel(Model):
  def __init__(self, X_train, y_train, train_scores, params={}):
    Model.__init__(self, X_train, y_train, train_scores, params)
    self.model = None    
  def fit(self):
    X_train = self.format_x(self.X_train)
    y_train = self.format_y(self.y_train)
    self.model = self.get_model()
    self.model.fit(X_train, y_train, verbose=False)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict_proba(self.format_x(X)))
    predictions.columns = [0,1,2]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)
  def get_model(self):
    iterations = self.params['iterations'] if 'iterations' in self.params else 1000
    devices = self.params['devices'] if 'devices' in self.params else '0:1'
    return CatBoostClassifier(iterations=iterations, devices=devices)