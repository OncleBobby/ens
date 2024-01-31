import pandas, numpy, sklearn
from importlib import import_module
from .model import Model

class StackingModel(Model):
  def __init__(self, X_train, y_train, train_scores, params={}):
    Model.__init__(self, X_train.replace({numpy.nan:0}), y_train, train_scores, params)
    self.model = None
    self.name = 'stacking_classifier'
  def fit(self):
    self.model = self.get_model()
    y_train = self.format_y(self.y_train)
    self.model.fit(self.X_train, y_train)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict_proba(X.replace({numpy.nan:0})))
    predictions.columns = [0,1,2]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)
  def get_model(self):
    estimators = [(name, estimator.get_model()) for name, estimator in self.params['estimators'].items()]
    print(f'estimators={estimators}')
    return sklearn.ensemble.StackingClassifier(estimators = estimators)
