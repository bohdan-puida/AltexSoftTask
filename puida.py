import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sys import argv

_, path_to_train, path_to_unlabeled_data, path_to_test, path_to_predictions = argv

import numpy as np
import pandas as pd

train = pd.read_csv(path_to_train)
test = pd.read_csv(path_to_test)
unlabeled = pd.read_csv(path_to_unlabeled_data)
imp = Imputer()





from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))



train = train.sample(frac=1)

test = test.sample(frac=1)





train[train == "na"] = np.nan





test[test == "na"] = np.nan




unlabeled[unlabeled == "na"] = np.nan

train = train.drop(['rowc', 'colc', 'ra', 'dec'], axis=1)

test = test.drop(['rowc', 'colc', 'ra', 'dec'], axis=1)


unlabeled = unlabeled.drop(['rowc', 'colc', 'ra', 'dec'], axis=1)


data = train.iloc[:, 1:39]

unlabeled = unlabeled.drop('objid', axis=1)



y = train.iloc[:, 39]

def norm_to_gauss(df):
    return (df - df.mean()) * 1.0 / (df.var())


data = data.apply(np.float64)

data_test = test.iloc[:, 1:39]

data_test = data_test.apply(np.float64)

unlabeled = unlabeled.apply(np.float64)

result =  pd.concat([data, unlabeled, data_test])

result = result.apply(norm_to_gauss)

result = imp.fit_transform(result)





result.shape




type(result)




data = result[:30000, :]




data.shape





unlabeled = result[30000:53333, :]





unlabeled.shape



data_test = result[53333:76666,: ]



data_test.shape




def argmax(preds):
    return np.asarray([np.argmax(line) for line in preds])



from keras.utils.np_utils import to_categorical   

cl = to_categorical(y, num_classes=3)




from keras.layers.advanced_activations import *




from keras import regularizers




from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(128,  input_dim=38, activation=PReLU()))
model.add(Dropout(0.4))
model.add(Dense(128, activation=PReLU()))
model.add(Dropout(0.3))
model.add(Dense(256, activation=PReLU()))
model.add(Dropout(0.3))
model.add(Dense(1024,activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(256,activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(128,activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(64,activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(9,  activation=PReLU()))
model.add(Dense(3,  activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=[f1])
hist = model.fit(data, cl, epochs=50, batch_size=1000)

predictions = argmax(model.predict(unlabeled))






xg = [*y, *predictions]




result = np.concatenate((data, unlabeled), axis=0)




import lightgbm as lgb

dtrain = lgb.Dataset(result, label=xg)






parameters = { 
    'boosting': 'dart',
    'objective': 'softmax',
    'num_class': 3, 
    'max_depth': 50,
    'num_leaves': 70,
    'bagging_freq': 1000,
    'bagging_fraction': 0.8,
    'bagging_seed': 42,
    'feature_fraction': 0.8,
    'drop_rate': 0,
    'max_drop': 10,
    'uniform_drop': True,
    'xgboost_dart_mode': True
    }  
num_round = 300





bst = lgb.train(parameters, dtrain, num_round)





preds = bst.predict(data_test)





yhat = argmax(preds)


df = pd.DataFrame(data={'objid': test.iloc[:, 0], 'prediction': yhat})
df.to_csv(path_to_predictions, encoding='utf-8', index=False)
