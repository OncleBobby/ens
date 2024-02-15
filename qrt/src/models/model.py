import numpy, logging, pandas
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
  def predict_proba(self, X):
    return pandas.DataFrame(self.model.predict_proba(self.format_x(X)))  
  def evaluate(self, X):
    predictions = self.predict(X)
    target = self.train_scores.loc[X.index].copy()
    return numpy.round(accuracy_score(predictions, target), 4)
  def save(self, X, root_path='../data/output/predictions/'):
    predictions = self.predict(X)
    predictions.columns = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
    predictions.index = X.index
    submission = predictions.reset_index()
    submission.to_csv(f'{root_path}{self.name}.csv', index=False)
  def save_proba(self, X, root_path='../data/output/mix/train/'):
    predictions = self.predict_proba(X)
    predictions.columns = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
    predictions.index = X.index
    submission = predictions.reset_index()
    submission.to_csv(f'{root_path}{self.name}.csv', index=False)
  def get_feature_importances(self):
    logging.info(f'Model.get_feature_importances ....')
    pass
  def format_x(self, x):
    return x.fillna(0)
  def format_y(self, y):
    z = y.copy()
    z['target'] = (z['HOME_WINS'] * 0 + z['DRAW'] * 1 + z['AWAY_WINS'] * 2).astype('category')
    return z['target']