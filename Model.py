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
import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
# Larger CNN for the MNIST Dataset
import numpy
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
from keras.utils import multi_gpu_model

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
        vae = Model(x, y)
        return vae
        
        
    def fit(dim, std, epochs, batch_size):
        
        vae = VAE.build(dim,std)
        vae.compile(optimizer= Adam(), loss= None)
        checkpointer= ModelCheckpoint(filepath= models_dir + '/[VAE]Lat_Dim={} | STD={}.hdf5'.format(dim, std),verbose=1,
                                  save_best_only=True)
        history = vae.fit(x_train_vae,
                                shuffle=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[checkpointer],
                                validation_data=(x_test_vae, None))
        
        import copy
        history_new = copy.deepcopy(history.history)
        del history_new['loss'][:1]
        del history_new['val_loss'][:1]
        import matplotlib.pyplot as plt
        '''
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/VAE" +'/[Dim {} | STD {}]Loss_per_Epoch.png'.format(dim, std))
        plt.close()
        '''
        df = pd.DataFrame(data = history_new, columns = [x for x in history_new])
        df.to_csv(lossEpoch_dir + "/[VAE]Loss_Epoch||STD={}||Dim={}".format(std,dim), index=False)

        

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
        cvae = Model(input_img, y)
        
        #cvae.compile(optimizer= Adam(), loss= None)


        return cvae
        
        
    def fit(dim, std, epochs, batch_size):
        #cx_train = x_train.reshape(x_train.shape[0],8,1).astype('float32')
        #cx_test = x_test.reshape(x_test.shape[0], 8,1).astype('float32')
        cvae = CVAE.build(dim,std)
        cvae.compile(optimizer= Adam(), loss= None)
        checkpointer= ModelCheckpoint(filepath= models_dir + '/[CVAE]Lat_Dim={} | STD={}.hdf5'.format(dim, std),verbose=1,
                                  save_best_only=True)
        history = cvae.fit(cx_train_vae,
                                shuffle=False,
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[checkpointer],
                                validation_data=(cx_test_vae, None))
        
        import copy
        history_new = copy.deepcopy(history.history)
        del history_new['loss'][:1]
        del history_new['val_loss'][:1]
        import matplotlib.pyplot as plt
        '''
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/CVAE" +'/[Dim {} | STD {}]Loss_per_Epoch.png'.format(dim, std))
        plt.close()
        '''
        df = pd.DataFrame(data = history_new, columns = [x for x in history_new])
        df.to_csv(lossEpoch_dir + "/[CVAE]Loss_Epoch||STD={}||Dim={}".format(std,dim), index=False)
        
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

        autoencoder = Model(input_img, decoded)
        #autoencoder = multi_gpu_model(autoencoder, gpus=5)
        
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
                                verbose=1,
                                validation_data=(x_test, x_test))
        
        import copy
        history_new = copy.deepcopy(history.history)
        del history_new['loss'][:1]
        del history_new['val_loss'][:1]
        import matplotlib.pyplot as plt
        '''
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/AE" +'/[Dim {}]Loss_per_Epoch.png'.format(dim))
        plt.close()
        '''
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

        autoencoder = Model(input_img, decoded)
        print(autoencoder.summary())

        #autoencoder = multi_gpu_model(autoencoder, gpus=5)
        return autoencoder

    def fit(dim, std, epochs, batch_size):
        #cx_train = x_train.reshape(x_train.shape[0],8,1).astype('float32')
        #cx_test = x_test.reshape(x_test.shape[0], 8,1).astype('float32')
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
        '''
        plt.plot(history_new['loss'])
        plt.plot(history_new['val_loss'])
        plt.title('Loss/Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Loss(train)', 'Loss(test)'], loc='upper left')
        plt.savefig(plots_dir + "/CAE" +'/[Dim {}]Loss_per_Epoch.png'.format(dim))
        plt.close()
        '''
        
        df = pd.DataFrame(data = history_new, columns = [x for x in history_new])
        df.to_csv(lossEpoch_dir + "/[CAE]Loss_Epoch||Dim={}".format(dim), index=False)
        
        
        

class AAE:
    def build(dim, std, mode):
        input_dim = x_train.shape[1]
        n_l1 = 6
        n_l2 = 6
        z_dim = dim
        std = std
        batch_size = 10000
        n_epochs = 10000
        learning_rate = 0.0005
        beta1 = 0.9
        results_path = './Results/Adversarial_Autoencoder'
        model_name = "[AAE]Lat_Dim={} | STD={}".format(z_dim, std)
        mode = mode

        if not os.path.exists(models_dir + "/" + model_name):
            os.makedirs(models_dir + "/" + model_name)

        tf.reset_default_graph()

        # Placeholders for input data and the targets
        x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
        x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
        real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
        decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')


        def form_results():
            """
            Forms folders for each run to store the tensorboard files, saved models and the log files.
            :return: three string pointing to tensorboard, saved models and log paths respectively.
            """
            folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_Adversarial_Autoencoder". \
                format(datetime.datetime.now(), z_dim, learning_rate, batch_size, n_epochs, beta1)
            tensorboard_path = results_path + folder_name + '/Tensorboard'
            saved_model_path = results_path + folder_name + '/Saved_models/'
            log_path = results_path + folder_name + '/log'
            if not os.path.exists(results_path + folder_name):
                os.mkdir(results_path + folder_name)
                os.mkdir(tensorboard_path)
                os.mkdir(saved_model_path)
                os.mkdir(log_path)
            return tensorboard_path, saved_model_path, log_path

        '''
        def generate_image_grid(sess, op):
            """
            Generates a grid of images by passing a set of numbers to the decoder and getting its output.
            :param sess: Tensorflow Session required to get the decoder output
            :param op: Operation that needs to be called inorder to get the decoder output
            :return: None, displays a matplotlib window with all the merged images.
            """
            x_points = np.arange(-10, 10, 1.5).astype(np.float32)
            y_points = np.arange(-10, 10, 1.5).astype(np.float32)

            nx, ny = len(x_points), len(y_points)
            plt.subplot()
            gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

            for i, g in enumerate(gs):
                z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
                z = np.reshape(z, (1, 2))
                x = sess.run(op, feed_dict={decoder_input: z})
                ax = plt.subplot(g)
                img = np.array(x.tolist()).reshape(28, 28)
                ax.imshow(img, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('auto')
            plt.show()
        '''

        def dense(x, n1, n2, name):
            """
            Used to create a dense layer.
            :param x: input tensor to the dense layer
            :param n1: no. of input neurons
            :param n2: no. of output neurons
            :param name: name of the entire dense layer.i.e, variable scope name.
            :return: tensor with shape [batch_size, n2]
            """
            with tf.variable_scope(name, reuse=None):
                weights = tf.get_variable("weights", shape=[n1, n2],
                                          initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
                bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
                out = tf.add(tf.matmul(x, weights), bias, name='matmul')
                return out


        # The autoencoder network
        def encoder(x, reuse=False):
            """
            Encode part of the autoencoder.
            :param x: input to the autoencoder
            :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
            :return: tensor which is the hidden latent variable of the autoencoder.
            """
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.name_scope('Encoder'):
                e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
                e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
                latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
                return latent_variable


        def decoder(x, reuse=False):
            """
            Decoder part of the autoencoder.
            :param x: input to the decoder
            :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
            :return: tensor which should ideally be the input given to the encoder.
            """
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.name_scope('Decoder'):
                d_dense_1 = tf.nn.relu(dense(x, z_dim, n_l2, 'd_dense_1'))
                d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
                output = dense(d_dense_2, n_l1, input_dim, 'd_output')
                return output


        def discriminator(x, reuse=False):
            """
            Discriminator that is used to match the posterior distribution with a given prior distribution.
            :param x: tensor of shape [batch_size, z_dim]
            :param reuse: True -> Reuse the discriminator variables,
                          False -> Create or search of variables before creating
            :return: tensor of shape [batch_size, 1]
            """
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.name_scope('Discriminator'):
                dc_den1 = tf.nn.relu(dense(x, z_dim, n_l1, name='dc_den1'))
                dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_den2'))
                output = dense(dc_den2, n_l2, 1, name='dc_output')
                return output


        def train(train_model=True):
            """
            Used to train the autoencoder by passing in the necessary inputs.
            :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
            :return: does not return anything
            """
            with tf.variable_scope(tf.get_variable_scope()):
                encoder_output = encoder(x_input)
                decoder_output = decoder(encoder_output)

            with tf.variable_scope(tf.get_variable_scope()):
                d_real = discriminator(real_distribution)
                d_fake = discriminator(encoder_output, reuse=True)

            with tf.variable_scope(tf.get_variable_scope()):
                decoder_image = decoder(decoder_input, reuse=True)

            # Autoencoder loss
            #autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))
            autoencoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x_target, logits=decoder_output))

            # Discrimminator Loss
            dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
            dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
            dc_loss = dc_loss_fake + dc_loss_real

            # Generator loss
            generator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

            all_variables = tf.trainable_variables()
            dc_var = [var for var in all_variables if 'dc_' in var.name]
            en_var = [var for var in all_variables if 'e_' in var.name]

            # Optimizers
            autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                           beta1=beta1).minimize(autoencoder_loss)
            discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                             beta1=beta1).minimize(dc_loss, var_list=dc_var)
            generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                         beta1=beta1).minimize(generator_loss, var_list=en_var)

            init = tf.global_variables_initializer()

            # Reshape immages to display them
            #input_images = tf.reshape(x_input, [-1, 28, 28, 1])
            #generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

            # Tensorboard visualization
            tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
            tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
            tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
            tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
            tf.summary.histogram(name='Real Distribution', values=real_distribution)
            #tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
            #tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
            summary_op = tf.summary.merge_all()

            # Saving the model
            saver = tf.train.Saver()
            step = 0
            with tf.Session() as sess:
                if train_model:
                    #tensorboard_path, saved_model_path, log_path = form_results()
                    sess.run(init)
                    #writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
                    #x_train = x_train[:-52]
                    #Early Stopping
                    checkpoints = []
                    n_checks = 4
                    checks = []
                    flag = 200
                    for check in range(n_checks):
                        if check == 0:
                            checks.append(0)
                        else:
                            checks.append((check * (int(flag/n_checks))) - 1)
                    for i in range(n_epochs):
                        
                        #np.random.shuffle(data_normalized)
                        train_data = data_normalized[:-1952]
                        x_train = train_data[:int(train_data.shape[0] * 0.9)]
                        x_test = train_data[int(train_data.shape[0] * 0.9):]
                        n_batches = int(x_train.shape[0] / batch_size)
                        #print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                        for b in range(0, n_batches):
                            z_real_dist = np.random.randn(batch_size, z_dim) * std
                            batch_x = x_train[b*batch_size:(b+1)*batch_size]
                            sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                            sess.run(discriminator_optimizer,
                                     feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})
                            sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                            if b % 50 == 0:
                                a_loss, d_loss, g_loss, summary = sess.run(
                                    [autoencoder_loss, dc_loss, generator_loss, summary_op],
                                    feed_dict={x_input: batch_x, x_target: batch_x,
                                               real_distribution: z_real_dist})
                                #writer.add_summary(summary, global_step=step)
                        print("Epoch: {} | Autoencoder Loss: {} | Discriminator Loss: {} | Generator Loss: {} ".format(i+1, 
                                                                                                                       a_loss, 
                                                                                                                       d_loss, 
                                                                                                                       g_loss))
                        #print("Autoencoder Loss: {}".format(a_loss))
                        #print("Discriminator Loss: {}".format(d_loss))
                        #print("Generator Loss: {}".format(g_loss))
                        log_path = lossEpoch_dir
                        with open(log_path + '/[AAE]Loss_Epoch||Dim={}||STD={}'.format(z_dim, std), 'a') as log:
                            #log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("{}".format(a_loss))
                            log.write(",{}".format(d_loss))
                            log.write(",{}\n".format(g_loss))
                        step += 1

                        saver.save(sess, save_path=models_dir + "/" + model_name + "/", global_step=step)
                        
                        checkpoints.append([a_loss, d_loss, g_loss])
                        if len(checkpoints) > flag:
                            checkpoints.pop(0)
                            
                        if (i+1) >= flag:
                          
                            safe_to_stop = []
                            need_to_reset = []
                            for checkpoint in range(len(checkpoints[0])):
                                for index in checks:
                                    losses_differences = np.absolute(checkpoints[flag - 1][checkpoint] - checkpoints[index][checkpoint])
                                    #module_derivative = np.absolute(top_part_of_derivative / (flag - index + 1))
                                    if ((i+1) % flag) == 0:
                                        print(losses_differences)
                                    if losses_differences <= 0.03:
                                        safe_to_stop.append(True)
                                    else:
                                        safe_to_stop.append(False)
                            for index in checks:
                                difference_between_losses = np.absolute(checkpoints[flag - 1 - index][2] 
                                                                      - checkpoints[flag - 1 - index][1])
                                if difference_between_losses >= 8.5:
                                    need_to_reset.append(True)
                                else:
                                    need_to_reset.append(False)
                            
                            if ((i+1) % 20) == 0:
                                print(safe_to_stop)
                                print(need_to_reset)
                                
                            #if False not in safe_to_stop:
                            #    return print("***Early Stopping***")
                            
                            if False not in need_to_reset:
                                print("***Reset Training***")
                                import os
                                os.remove(lossEpoch_dir + "/[AAE]Loss_Epoch||Dim={}||STD={}".format(z_dim, std))
                                this_model_dir = models_dir + "/[AAE]Lat_Dim={} | STD={}".format(z_dim, std)
                                arquivos = os.listdir(this_model_dir)
                                for arquivo in arquivos:
                                    os.remove(this_model_dir + "/" + arquivo)
                                return 0
                    
                else:
                    # Get the latest results folder
                    #all_results = os.listdir(results_path)
                    #all_results.sort()
                    import os
                    saver.restore(sess, save_path=tf.train.latest_checkpoint(models_dir + "/" + model_name + "/"))
                    #generate_image_grid(sess, op=decoder_image)

                    zeros = np.zeros([19,data.shape[1]])
                    data_ad = np.concatenate([data_normalized,zeros],axis=0)
                    n_batches = int(data_ad.shape[0] / batch_size)
                        #print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                    for b in range(0, n_batches):
                        batch_x = data_ad[b*batch_size:(b+1)*batch_size]
                        if b == 0:
                            predict = sess.run(encoder_output, feed_dict={x_input: batch_x})
                        else:
                            predict_tmp = sess.run(encoder_output, feed_dict={x_input: batch_x})
                            predict = np.concatenate([predict, predict_tmp], axis=0)
                          #if b % 100000:
                          #  print(b)
                    predict = predict[:-19,:]
                    df = pd.DataFrame(data = predict)
                    df.to_csv(latentSpace_dir + "/" + f"[AAE]Lat_Dim={lat_dim} | STD={std}.csv", index=False)

                    print("***Finalizado***")

        if mode == "train":
            status = train(train_model=True)
            if status == 0: 
                return 0
            else:
                return 1
        else:
            train(train_model=False)
