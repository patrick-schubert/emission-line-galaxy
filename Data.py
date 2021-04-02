import pandas as pd
import numpy as np

data_normalized = pd.read_csv("Training_Data")
data_normalized = np.asarray(data_normalized.values[:,:-2])

#np.random.shuffle(data_normalized)

x_train = data_normalized[:int(data_normalized.shape[0] * 0.9)]
x_test = data_normalized[int(data_normalized.shape[0] * 0.9):]

x_train_vae = x_train[:-152,:]
x_test_vae = x_test[:-129,:]

cx_train = x_train.reshape(x_train.shape[0],data_normalized.shape[1],1).astype('float32')
cx_test = x_test.reshape(x_test.shape[0], data_normalized.shape[1],1).astype('float32')

cx_train_vae = cx_train[:-152, :]
cx_test_vae = cx_test[:-129, :]

data_normalized_vec = data_normalized.reshape(data_normalized.shape[0], data_normalized.shape[1],1).astype('float32')