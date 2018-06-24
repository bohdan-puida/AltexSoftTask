from sys import argv

_, path_to_train, path_to_unlabeled_data, path_to_test, path_to_predictions = argv

import numpy as np
import pandas as pd

train = pd.read_csv(path_to_train)
test = pd.read_csv(path_to_test)

train = train.sample(frac=1)

train[train == 'na'] = np.nan
test[test == 'na'] = np.nan

train = train.drop(['rowc', 'colc', 'ra', 'dec'], axis=1)
test = test.drop(['rowc', 'colc', 'ra', 'dec'], axis=1)

data = train.iloc[:, 1:39]
y = train.iloc[:, 39]

def norm_to_gauss(df):
    return (df - df.mean()) * 1.0 / (df.var())

data = data.apply(np.float64)

data = data.apply(norm_to_gauss)

data_test = test.iloc[:, 1:39]
data_test = data_test.apply(np.float32)
data_test = data_test.apply(norm_to_gauss)

import sklearn
from sklearn.impute import MICEImputer
scaler = MICEImputer(n_imputations=10)
scaler.fit(data)
data = scaler.transform(data)
data_test = scaler.transform(data_test)


def argmax(preds):
    return np.asarray([np.argmax(line) for line in preds])

from keras.utils.np_utils import to_categorical   

cl = to_categorical(y, num_classes=3)


from keras.layers.advanced_activations import *

from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(256, init='uniform', input_dim=38, activation=PReLU()))
model.add(Dropout(0.5))
model.add(Dense(128, init='uniform', activation=PReLU()))
model.add(Dropout(0.3))
model.add(Dense(64, init='uniform', activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(128, init='uniform', activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(64, init='uniform', activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(128, init='uniform', activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(64, init='uniform', activation=PReLU()))
model.add(Dropout(0.2))
model.add(Dense(9, init='uniform', activation=PReLU()))
model.add(Dense(3, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])
hist = model.fit(data, cl, epochs=15, batch_size=500, verbose=0)

predictions = argmax(model.predict(data_test))

df = pd.DataFrame(data={'objid': test.iloc[:, 0], 'prediction': predictions})
df.to_csv(path_to_predictions, encoding='utf-8', index=False)
