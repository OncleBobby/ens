import pandas, xgboost
from .model import Model

class XgboostClassifierModel(Model):
  def __init__(self, X_train, y_train, train_scores, params):
    Model.__init__(self, X_train, y_train, train_scores, params)
    self.model = None    
  def get_model(self):
    params = self.params.copy()
    return xgboost.XGBClassifier(**params)
  def fit(self):
    self.model = self.get_model()
    self.model.fit(self.X_train, self.format_y(self.y_train))
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict_proba(self.format_x(X)))
    predictions.columns = [0,1,2]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)
  def get_feature_importances(self):
    feature_names = [self.X_train.columns[i] for i in range(self.X_train.shape[1])]
    df_importances = pandas.DataFrame({'feature': feature_names, 'importance': self.model.feature_importances_})
    return df_importances.sort_values(by=['importance'], ascending=False)
  