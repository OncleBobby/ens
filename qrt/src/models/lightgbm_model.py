import pandas, lightgbm, logging, numpy
from sklearn import model_selection
from .model import Model

class LightgbmModel(Model):
  def __init__(self, X_train, y_train, train_scores, params):
    Model.__init__(self, X_train.replace({numpy.nan:0}), y_train, train_scores, params)
    self.model = None
  def get_model(self):
    params = self.params.copy()
    return lightgbm.LGBMClassifier(**params)
  def fit(self):
    params = self.params.copy()
    train_size=0.8
    random_state=42
    y_train = self.format_y(self.y_train)
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(self.X_train, y_train, train_size=train_size, random_state=random_state)
    eval_set = [(X_valid, y_valid),(X_train, y_train)]
    self.model = self.get_model()
    self.model.fit(X_train, y_train, eval_set=eval_set)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict_proba(self.format_x(X)))
    predictions.columns = [0,1,2]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)
  def get_feature_importances(self):
    feature_names = [self.X_train.columns[i] for i in range(self.X_train.shape[1])]
    df_importances = pandas.DataFrame({'feature': feature_names, 'importance': self.model.feature_importances_})
    return df_importances.sort_values(by=['importance'], ascending=False)
  