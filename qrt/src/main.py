import pandas, time, numpy, yaml, logging, logging.config, data_access
from data_access import get_X, get_y, get_train_test
from models.factory import ModelFactory

with open('qrt/confs/logs.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)
root_path = 'qrt'

def evalute_models(X_train, y_train, X_test):
    save_model = True
    with open('qrt/confs/models.yaml', 'r') as file:
        configurations = yaml.safe_load(file)
    factory = ModelFactory(configurations, X_train, y_train, train_scores)
    lines = []
    for model in factory.get_models():
        model.train()
        lines.append(eval_model(model, save_model))
    df = pandas.DataFrame(lines)
    df = df.sort_values(by=['score'], ascending=False)
def eval_model(model, save_model=False):
    start = time.time()
    model.train()
    score = model.evaluate(X_test)
    end = time.time()
    logging.info(f'{model.name}={score} in {numpy.round((end-start), 2)}s')
    if save_model:
        model.save(test_data, root_path)
    return {'name': model.name, 'score': score, 'time': numpy.round((end-start), 2)}

data_access.root_path = root_path
feature=['HOME_WINS', 'DRAW', 'AWAY_WINS']
train_data = get_X('train')
train_scores = get_y('train')
test_data = get_X('test')
X_train, y_train, X_test, y_test, target = get_train_test(train_size=0.8, random_state=42, feature=feature)
evalute_models(X_train, y_train, X_test)
