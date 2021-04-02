import pandas as pd
import numpy as np
#from scipy.stats import norm
import os
from Data import *
from Dirs import *
from keras.layers import Input, Dense, Lambda, Layer, Reshape, Flatten
from keras.layers import Conv1D, UpSampling1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras import metrics
import numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Reshape
from keras.layers import Embedding
from keras.layers import Conv1D, UpSampling1D
from keras.layers import MaxPooling2D, MaxPooling1D, Input, BatchNormalization, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

class VAE:
    original_dim = x_train.shape[1]
    def build(dim, std):
        original_dim = VAE.original_dim
        latent_dim = dim
        epsilon_std = std
     
        intermediate_dim = int(np.ceil((original_dim + latent_dim) / 2))
        
       
        x = Input(shape=(original_dim,))
        h = Dense(intermediate_dim, activation='relu')(x)
        h = Dense(intermediate_dim, activation='relu')(x)
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)


        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                    stddev=epsilon_std)
            #vec0 = np.random.normal(-15.0, epsilon_std, (200,1))
            #vec1 = np.random.normal(-10.0, epsilon_std, (200,1))
            #vec2 = np.random.normal(-5.0, epsilon_std, (200,1))
            #vec3 = np.random.normal(0.0, epsilon_std, (200,1))
            #vec4 = np.random.normal(5.0, epsilon_std, (200,1))
            #vec5 = np.random.normal(10.0, epsilon_std, (200,1))
            #vec6 = np.random.normal(15.0, epsilon_std, (200,1))
            #vec7 = np.random.normal(20.0, epsilon_std, (200,1))
            #vec8 = np.random.normal(25.0, epsilon_std, (200,1))
            #vec9 = np.random.normal(30.0, epsilon_std, (200,1))
            #vec = [vec0,vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,vec9]
            #vec = [vec0,vec1]
            #vecfinal = vec0[:,0]
            #vecfinal = np.reshape(vecfinal, (vecfinal.shape[0],1))
            
            #for i in range(1,latent_dim):
            #    tmp = np.reshape(vec[i][:,0], (vec[i].shape[0],1))
            #    vecfinal = np.concatenate([vecfinal, tmp] ,axis=1)
            #epsilon = tf.convert_to_tensor(vecfinal, dtype=tf.float32)
            return z_mean + K.exp(z_log_var) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later

        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_h1 = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')

        h_decoded1 = decoder_h(z)
        h_decoded = decoder_h1(h_decoded1)
        x_decoded_mean = decoder_mean(h_decoded)


        # Custom loss layer
        class CustomVariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomVariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, x, x_decoded_mean):
                xent_loss =  metrics.mean_squared_error(x, x_decoded_mean)
                kl_loss = - 5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                return K.mean(xent_loss + kl_loss)

            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                loss = self.vae_loss(x, x_decoded_mean)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        y = CustomVariationalLayer()([x, x_decoded_mean])
        vae = Model(x, z)
        return vae
        
        
        
    def fit(dim, std, epochs, batch_size):
        
        vae = VAE.build(dim,std)
        vae.compile(optimizer= Adam(), loss= None)
        checkpointer= ModelCheckpoint(filepath= models_dir + '/[VAE]Lat_Dim={} | STD={}.hdf5'.format(dim, std),verbose=1,
                                  save_best_only=True)
        history = vae.fit(x_train,
                                shuffle=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[checkpointer],
                                validation_data=(x_test, None))
        
        import copy
        history_new = copy.deepcopy(history.history)
        del history_new['loss'][:1]
        del history_new['val_loss'][:1]
        import matplotlib.pyplot as plt
        
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/VAE" +'/[Dim {} | STD {}]Loss_per_Epoch.png'.format(dim, std))
        df = pd.DataFrame(data = history_new, columns = [x for x in history_new])
        df.to_csv(lossEpoch_dir + "/[VAE]Loss_Epoch||Dim={}||STD={}".format(dim,std), index=False)

        

class CVAE:
    original_dim = x_train.shape[1]
    def build(dim, std):
        original_dim = CVAE.original_dim
        #print(original_dim)
        latent_dim = dim
        epsilon_std = std
        input_img = Input(shape=(original_dim, 1))

        x = Conv1D(8, 3, padding='same')(input_img)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(16, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(32, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same')(x)


        flatten = Flatten()(x)
        dense1 = Dense(cx_train.shape[1])(flatten)
        z_mean = Dense(latent_dim)(dense1)
        z_log_var = Dense(latent_dim)(dense1)


        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                    stddev=epsilon_std)
            #vec0 = np.random.normal(-15.0, epsilon_std, (200,1))
            #vec1 = np.random.normal(-10.0, epsilon_std, (200,1))
            #vec2 = np.random.normal(-5.0, epsilon_std, (200,1))
            #vec3 = np.random.normal(0.0, epsilon_std, (200,1))
            #vec4 = np.random.normal(5.0, epsilon_std, (200,1))
            #vec5 = np.random.normal(10.0, epsilon_std, (200,1))
            #vec6 = np.random.normal(15.0, epsilon_std, (200,1))
            #vec7 = np.random.normal(20.0, epsilon_std, (200,1))
            #vec8 = np.random.normal(25.0, epsilon_std, (200,1))
            #vec9 = np.random.normal(30.0, epsilon_std, (200,1))
            #vec = [vec0,vec1,vec2,vec3,vec4,vec5,vec6,vec7,vec8,vec9]
            #vec = [vec0,vec1]
            #vecfinal = vec0[:,0]
            #vecfinal = np.reshape(vecfinal, (vecfinal.shape[0],1))
            
            #for i in range(1,latent_dim):
            #    tmp = np.reshape(vec[i][:,0], (vec[i].shape[0],1))
            #    vecfinal = np.concatenate([vecfinal, tmp] ,axis=1)
            #epsilon = tf.convert_to_tensor(vecfinal, dtype=tf.float32)
            return z_mean + K.exp(z_log_var) * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])
        dense2 = Dense(cx_train.shape[1])(z)
        reshaped = Reshape((int(x.shape[1]), int(cx_train.shape[1]/int(x.shape[1]))))(dense2)


        x = Conv1D(32, 3, padding='same')(reshaped)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(16, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(8, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2)(x)
        x_decoded_mean = Conv1D(1, 1, activation='sigmoid', padding='same')(x)
    
    
        # Custom loss layer
        class CustomVariationalLayer(Layer):
            def __init__(self, **kwargs):
                self.is_placeholder = True
                super(CustomVariationalLayer, self).__init__(**kwargs)

            def vae_loss(self, input_img, x_decoded_mean):
                #print(input_img.shape,x_decoded_mean.shape)
                #xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)[:,0]
                xent_loss = K.mean(metrics.mean_squared_error(input_img, x_decoded_mean),axis=1)
                kl_loss = - 5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                #kl_loss = - 0.5 * (1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))        #print(kl_loss.shape)
                #print(xent_loss.shape,kl_loss.shape)
                return K.mean(xent_loss + kl_loss)

                #return kl_loss
            def call(self, inputs):
                x = inputs[0]
                x_decoded_mean = inputs[1]
                loss = self.vae_loss(input_img, x_decoded_mean)
                self.add_loss(loss, inputs=inputs)
                # We won't actually use the output.
                return x

        y = CustomVariationalLayer()([input_img, x_decoded_mean])
        cvae = Model(input_img, z)
        
        #cvae.compile(optimizer= Adam(), loss= None)


        return cvae
        
        
    def fit(dim, std, epochs, batch_size):
        #cx_train = x_train.reshape(x_train.shape[0],8,1).astype('float32')
        #cx_test = x_test.reshape(x_test.shape[0], 8,1).astype('float32')
        cvae = CVAE.build(dim,std)
        cvae.compile(optimizer= Adam(), loss= None)
        checkpointer= ModelCheckpoint(filepath= models_dir + '/[CVAE]Lat_Dim={} | STD={}.hdf5'.format(dim, std),verbose=1,
                                  save_best_only=True)
        history = cvae.fit(cx_train,
                                shuffle=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[checkpointer],
                                validation_data=(cx_test, None))
        
        import copy
        history_new = copy.deepcopy(history.history)
        del history_new['loss'][:1]
        del history_new['val_loss'][:1]
        import matplotlib.pyplot as plt
        
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/CVAE" +'/[Dim {} | STD {}]Loss_per_Epoch.png'.format(dim, std))
        df = pd.DataFrame(data = history_new, columns = [x for x in history_new])
        df.to_csv(lossEpoch_dir + "/[CVAE]Loss_Epoch||Dim={}||STD={}".format(dim,std), index=False)
        
class AE:
    original_dim = x_train.shape[1]
    def build(dim):
        original_dim = CAE.original_dim
        latent_dim = dim
        intermediate_dim = int(np.ceil((original_dim + latent_dim) / 2))
        
        input_img = Input(shape=(original_dim, ))

        x = Dense(intermediate_dim, activation='relu')(input_img)    
        encoded = Dense(latent_dim)(x)



        x = Dense(intermediate_dim, activation='relu')(encoded)
        decoded = Dense(original_dim, activation='sigmoid')(x)

        autoencoder = Model(input_img, encoded)
        
        return autoencoder
    
    def fit(dim, std, epochs, batch_size):
        
        ae = AE.build(dim)
        ae.compile(optimizer= Adam(), loss= 'mse')
        checkpointer= ModelCheckpoint(filepath= models_dir + '/[AE]Lat_Dim={}.hdf5'.format(dim),verbose=1,
                                  save_best_only=True)
        
        history = ae.fit(x_train,x_train,
                                shuffle=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[checkpointer],
                                validation_data=(x_test, x_test))
        
        import copy
        history_new = copy.deepcopy(history.history)
        del history_new['loss'][:1]
        del history_new['val_loss'][:1]
        import matplotlib.pyplot as plt
        
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/AE" +'/[Dim {}]Loss_per_Epoch.png'.format(dim))
       
        df = pd.DataFrame(data = history_new, columns = [x for x in history_new])
        df.to_csv(lossEpoch_dir + "/[AE]Loss_Epoch||Dim={}".format(dim), index=False)
        
class CAE:
    original_dim = x_train.shape[1]
    def build(dim):
        original_dim = CAE.original_dim
        #print(original_dim)
        latent_dim = dim
        input_img = Input(shape=(original_dim, 1))

        x = Conv1D(8, 3, padding='same')(input_img)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(16, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(32, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(2, padding='same')(x)


        flatten = Flatten()(x)
        dense1 = Dense(cx_train.shape[1])(flatten)
        encoded = Dense(latent_dim)(dense1)
        dense2 = Dense(cx_train.shape[1])(encoded)
        reshaped = Reshape((int(x.shape[1]), int(cx_train.shape[1]/int(x.shape[1]))))(dense2)


        x = Conv1D(32, 3, padding='same')(reshaped)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(16, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(8, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, 1, activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, encoded)

        return autoencoder

    def fit(dim, std, epochs, batch_size):
        cx_train = x_train.reshape(x_train.shape[0],8,1).astype('float32')
        cx_test = x_test.reshape(x_test.shape[0], 8,1).astype('float32')
        cae = CAE.build(dim)
        cae.compile(optimizer= Adam(), loss= 'mse')
        checkpointer= ModelCheckpoint(filepath= models_dir + '/[CAE]Lat_Dim={}.hdf5'.format(dim),verbose=1,
                                  save_best_only=True)
        history = cae.fit(cx_train, cx_train,
                                shuffle=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[checkpointer],
                                validation_data=(cx_test, cx_test))
        
        import copy
        history_new = copy.deepcopy(history.history)
        del history_new['loss'][:1]
        del history_new['val_loss'][:1]
        import matplotlib.pyplot as plt
        
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/CAE" +'/[Dim {}]Loss_per_Epoch.png'.format(dim))
        
        
        df = pd.DataFrame(data = history_new, columns = [x for x in history_new])
        df.to_csv(lossEpoch_dir + "/[CAE]Loss_Epoch||Dim={}".format(dim), index=False)