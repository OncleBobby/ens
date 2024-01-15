import pandas, xgboost, numpy
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from .model import Model

class SklearnModel(Model):
  def __init__(self, X_train, y_train, train_scores, params={}):
    Model.__init__(self, X_train.replace({numpy.nan:0}), y_train, train_scores, params)
    self.model = None    
  def train(self):
    self.model = KNeighborsClassifier()
    self.model.fit(self.X_train, self.y_train)
  def predict(self, X):
    predictions = pandas.DataFrame(self.model.predict_proba(X.replace({numpy.nan:0})))


    print(f'predictions={predictions.head()}')

    predictions[2] = 0
    predictions.columns = [0,2,1]
    predictions = (predictions.reindex(columns=[0,1,2]).rank(1,ascending=False)==1).astype(int).values
    return pandas.DataFrame(predictions)


# # Split iris data in train and test data
# # A random permutation, to split the data randomly
# np.random.seed(0)
# indices = np.random.permutation(len(iris_X))
# iris_X_train = iris_X[indices[:-10]]
# iris_y_train = iris_y[indices[:-10]]
# iris_X_test = iris_X[indices[-10:]]
# iris_y_test = iris_y[indices[-10:]]
# # Create and fit a nearest-neighbor classifier
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier()
# knn.fit(iris_X_train, iris_y_train)
# knn.predict(iris_X_test)
# iris_y_test