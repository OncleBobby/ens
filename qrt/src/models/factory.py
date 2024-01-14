from importlib import import_module

class ModelFactory():
  def __init__(self, configurations, X_train, y_train, X_valid, y_valid, train_scores):
    self.configurations = configurations
    self.X_train = X_train
    self.y_train = y_train
    self.X_valid = X_valid
    self.y_valid = y_valid
    self.train_scores = train_scores
  def get_model(self, name):
    configuration = self.configurations[name]
    class_str = configuration['class']
    params = configuration['params']
    module_path, class_name = class_str.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)(self.X_train, self.y_train, self.X_valid, self.y_valid, self.train_scores, params)
