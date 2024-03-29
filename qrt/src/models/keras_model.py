import pandas, numpy, keras
from sklearn import model_selection
from .model import Model

class KerasModel(Model):
  def __init__(self, X_train, y_train, train_scores, params={}):
    Model.__init__(self, X_train.replace({numpy.nan:0}), y_train, train_scores, params)
    self.model = None    
  def fit(self):
    train_size=0.8
    random_state=42
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(self.X_train, self.y_train, train_size=train_size, random_state=random_state)

    layers = [keras.Input(shape=self.X_train.shape[1])]
    for layer in self.params['layers']:
      layers.append(keras.layers.Dense(layer['units'], activation=layer['activation']))
    layers.append(keras.layers.Dense(3, activation="sigmoid"))
    self.model = keras.Sequential(layers)
    self.model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
    history = self.model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size = 10, epochs = 100, verbose=0)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict(X))
    predictions = pandas.DataFrame((predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values)
    return predictions

