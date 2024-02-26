import pandas, numpy
from sklearn import model_selection

root_path = '..'

def read_statistics(name, type, way):
    return pandas.read_csv(f'{root_path}/data/input/statistics/{name}_{way}_{type}_statistics_df.csv', index_col=0, encoding='latin-1')
def get_X(name, type='team'):
    home = read_statistics(name, type, 'home', ).replace({numpy.inf:numpy.nan, -numpy.inf:numpy.nan})
    away = read_statistics(name, type, 'away').replace({numpy.inf:numpy.nan, -numpy.inf:numpy.nan})
    columns = list(home.columns)
    for column in ['LEAGUE', 'POSITION', 'TEAM_NAME', 'PLAYER_NAME']:
        if column in columns: columns.remove(column)
    if type == 'team':
        home = home.groupby(by=["ID"]).sum()
        away = away.groupby(by=["ID"]).sum()
    home = home[columns]
    away = away[columns]
    data =  home.iloc[:,2:] + away.iloc[:,2:] * -1
    return data
def get_train_test(train_size=0.8, random_state=42, type='team'):
    train_data = get_X('train', type)
    train_scores = get_y()
    train_new_y = train_scores[['HOME_WINS', 'DRAW', 'AWAY_WINS']]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(train_data, train_new_y, train_size=train_size, random_state=random_state)
    target = train_scores.loc[X_test.index].copy()
    return X_train, y_train, X_test, y_test, target
def get_y(name='train'):
    train_data = get_X(name)
    train_scores = pandas.read_csv(f'{root_path}/data/input/Y_train_1rknArQ.csv', index_col=0, encoding='latin-1')
    train_scores = train_scores.loc[train_data.index]
    return train_scores
