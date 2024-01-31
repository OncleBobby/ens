from importlib import import_module

class ModelFactory():
  def __init__(self, configurations, X_train, y_train, train_scores):
    self.configurations = configurations
    self.X_train = X_train
    self.y_train = y_train
    self.train_scores = train_scores
  def get_model(self, name, configuration = None):
    if configuration is None:
      configuration = self.configurations[name]
    class_str = configuration['class']
    params = configuration['params']
    module_path, class_name = class_str.rsplit('.', 1)
    module = import_module(module_path)
    model = getattr(module, class_name)(self.X_train, self.y_train, self.train_scores, params)
    model.name = name
    return model
  def get_models(self):
    models = []
    for name, configuration in self.configurations.items():
      models.append(self.get_model(name, configuration))
    return models