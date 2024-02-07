import pandas, xgboost, logging
from sklearn import model_selection
from .model import Model

class XgboostClassifierModel(Model):
  def __init__(self, X_train, y_train, train_scores, \
        params={
          'booster': 'gbtree',
          'tree_method':'hist',
          'max_depth': 8, 
          'learning_rate': 0.025,
          'objective': 'multi:softprob',
          'num_class': 3,
          'eval_metric':'mlogloss'
        }):
    Model.__init__(self, X_train, y_train, train_scores, params)
    self.model = None    
  def fit(self):
    params = self.params.copy()
    self.model = xgboost.XGBClassifier(
        booster=params['booster'],
        tree_method=params['tree_method'],
        max_depth=params['max_depth'],    
        learning_rate=params['learning_rate'],
        objective=params['objective'],
        num_class=params['num_class'],
        eval_metric=params['eval_metric']
    )
    self.model.fit(self.X_train, self.format_y(self.y_train))
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict_proba(self.format_x(X)))
    predictions.columns = [0,1,2]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)
  def get_feature_importances(self):
    logging.info(f'SklearnModel.get_feature_importances ....')
    feature_names = [self.X_train.columns[i] for i in range(self.X_train.shape[1])]


    logging.info(f'feature_importances_={dir(self.model)}')

    logging.info(f'feature_importances_={self.model.feature_importances_}')
    logging.info(f'feature_names_in_={self.model.feature_names_in_}')
    logging.info(f'feature_types={self.model.feature_types}')
    logging.info(f'importance_type={self.model.importance_type}')
    logging.info(f'base_score={self.model.base_score}')
    logging.info(f'best_iteration={self.model.best_iteration}')
    logging.info(f'best_score={self.model.best_score}')
    logging.info(f'best_iteration={self.model.best_iteration}')
    logging.info(f'best_iteration={self.model.best_iteration}')
    logging.info(f'best_iteration={self.model.best_iteration}')

    df_importances = pandas.DataFrame({'feature': feature_names, 'importance': self.model.feature_importances_})
    return df_importances.sort_values(by=['importance'], ascending=False)
  