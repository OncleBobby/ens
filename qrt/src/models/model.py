import numpy
from sklearn.metrics import accuracy_score

class Model:
  def __init__(self, X_train, y_train, X_valid, y_valid, train_scores):
    self.X_train = X_train
    self.y_train = y_train
    self.X_valid = X_valid
    self.y_valid = y_valid
    self.train_scores = train_scores
    self.name = self.__class__.__name__
  def train(self):
    pass
  def predict(self, X):
     pass
  def evaluate(self, X):
    predictions = self.predict(X)
    target = self.train_scores.loc[X.index].copy()
    return numpy.round(accuracy_score(predictions, target), 4)
  def save(self, X):
    predictions = self.predict(X)
    predictions.columns = ['HOME_WINS', 'DRAW', 'AWAY_WINS']
    predictions.index = X.index
    submission = predictions.reset_index()
    print(f'submission={submission}')
    submission.to_csv(f'./{self.name}_submission.csv', index=False)    