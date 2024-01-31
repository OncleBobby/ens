import pandas, numpy
from sklearn import model_selection

root_path = '..'

def read_statistics(name, type):
    return pandas.read_csv(f'{root_path}/data/{type}_Data/{type}_{name}_statistics_df.csv', index_col=0, encoding='latin-1')
def get_X(name):
    return get_X_2(name)
def get_y(name='train'):
    train_data = get_X(name)
    train_scores = pandas.read_csv(f'{root_path}/data/Y_train_1rknArQ.csv', index_col=0, encoding='latin-1')
    train_scores = train_scores.loc[train_data.index]
    return train_scores
def get_train_test(train_size=0.8, random_state=42):
    train_data = get_X('train')
    train_scores = get_y()
    train_new_y = train_scores[['HOME_WINS', 'DRAW', 'AWAY_WINS']]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(train_data, train_new_y, train_size=train_size, random_state=random_state)
    target = train_scores.loc[X_test.index].copy()
    return X_train, y_train, X_test, y_test, target
def get_X_1(name):
    home_team = read_statistics('home_team', name)
    away_team = read_statistics('away_team', name)
    if name == 'train':
        home_team = home_team.iloc[:,2:]
        away_team = away_team.iloc[:,2:]
    home_team.columns = 'HOME_' + home_team.columns
    away_team.columns = 'AWAY_' + away_team.columns
    data =  pandas.concat([home_team, away_team],join='inner',axis=1)
    return data.replace({numpy.inf:numpy.nan, -numpy.inf:numpy.nan})
def get_X_2(name):
    home_team = read_statistics('home_team', name).replace({numpy.inf:numpy.nan, -numpy.inf:numpy.nan})
    away_team = read_statistics('away_team', name).replace({numpy.inf:numpy.nan, -numpy.inf:numpy.nan})
    if name == 'train':
        home_team = home_team.iloc[:,2:]
        away_team = away_team.iloc[:,2:]
    data =  home_team.iloc[:,2:] + away_team.iloc[:,2:] * -1
    return data