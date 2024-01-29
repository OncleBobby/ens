import pandas, numpy
from importlib import import_module
from .model import Model

class SklearnModel(Model):
  def __init__(self, X_train, y_train, train_scores, params={}):
    Model.__init__(self, X_train.replace({numpy.nan:0}), y_train, train_scores, params)
    self.model = None    
  def fit(self):
    self.model = self._get_model()
    y_train = self._format_y(self.y_train)
    self.model.fit(self.X_train, y_train)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict_proba(X.replace({numpy.nan:0})))
    predictions.columns = [0,1,2]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)
  def _get_model(self):
    class_str = self.params['class_name']
    s = class_str.split('.')
    module_path = '.'.join(s[:-1])
    class_name = s[-1]
    module = import_module(module_path)
    return getattr(module, class_name)()
  def _format_y(self, y):
    z = y.copy()
    z['target'] = (z['HOME_WINS'] * 0 + z['DRAW'] * 1 + z['AWAY_WINS'] * 2).astype('category')
    return z['target']