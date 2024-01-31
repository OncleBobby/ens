import numpy
from sklearn.metrics import accuracy_score

class Model:
  def __init__(self, X_train, y_train, train_scores, params=None):
    self.X_train = X_train
    self.y_train = y_train
    self.train_scores = train_scores
    self.params = params
    self.name = self.__class__.__name__
  def fit(self):
    pass
  def predict(self, X):
     pass
  def evaluate(self, X):
    predictions = self.predict(X)
    target = self.train_scores.loc[X.index].copy()
    return numpy.round(accuracy_score(predictions, target), 4)
  def save(self, X, root_path='..'):
    predictions = self.predict(X)
    predictions.columns = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
    predictions.index = X.index
    submission = predictions.reset_index()
    submission.to_csv(f'{root_path}/data/predictions/{self.name}.csv', index=False)
  def format_x(self, x):
    return x.replace({0:numpy.nan})
  def format_y(self, y):
    z = y.copy()
    z['target'] = (z['HOME_WINS'] * 0 + z['DRAW'] * 1 + z['AWAY_WINS'] * 2).astype('category')
    return z['target']