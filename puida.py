from sys import argv

_, path_to_train, path_to_unlabeled_data, path_to_test, path_to_predictions = argv

import numpy as np
import pandas as pd

train = pd.read_csv(path_to_train)
test = pd.read_csv(path_to_test)

train = train.sample(frac=1)

train[train == 'na'] = np.nan
test[test == 'na'] = np.nan

data = train.iloc[:, 1:43]
y = train.iloc[:, 43]

def norm_to_gauss(df):
    return (df - df.mean()) * 1.0 / (df.var())

data = data.apply(np.float64)

data = data.apply(norm_to_gauss)

data_test = test.iloc[:, 1:43]
data_test = data_test.apply(np.float32)
data_test = data_test.apply(norm_to_gauss)

import sklearn
from sklearn.impute import MICEImputer
scaler = MICEImputer(n_imputations=70)
scaler.fit(data)
data = scaler.transform(data)
data_test = scaler.transform(data_test)


def argmax(preds):
    return np.asarray([np.argmax(line) for line in preds])

from keras.utils.np_utils import to_categorical   

cl = to_categorical(y, num_classes=3)


from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(128, init='uniform', input_dim=42, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(64, init='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(12, init='uniform', activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(data, cl, epochs=25, batch_size=100)
predictions = model.predict(data_train)

pr = argmax(predictions)
d = {'objid': train.iloc[:, 0], 'prediction': pr}
df = pd.DataFrame(data=d)
df.to_csv(path_to_predictions, encoding='utf-8', index=False)