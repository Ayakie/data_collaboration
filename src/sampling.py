# sorce: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/sampling.py

import numpy as np
from sklearn.preprocessing import normalize
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, Reshape, Input, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from options import args_parser

args = args_parser()

def make_anchors(X_train, nanc=args.nanc, anc_type=args.anc_type, hidden_dim=256, epochs=2000, latent_dim=100):
    '''
    Parameters
    --------------
    nanc: int
        number of anchor data
    X_train: 
        training data to generate anchor data
    anc_type: 'random', 'gan', 'saved', 'raw'
        'gan_new': generate anchor data by GAN from X_train to generate anchor data
        'gan': load saved weight of anchor generator previously trained by anc_type='gan_new'
        'raw': sample from original data(X_train) as anchors
    
    '''
    if anc_type == 'random':
    
        # anc = np.random.uniform(low=np.min(X_train), high=np.max(X_train), size=(nanc, X_train.shape[1]))
        # anc = normalize(anc)
        anc = np.random.uniform(0, 255, size=(nanc, X_train.shape[1]))
    
    elif anc_type == 'gan_new':

        gan = GAN(input_shape=X_train.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim)
        gan.train(X_train, epochs=epochs, batch_size=32)
        
        noise = np.random.normal(0, 1, size=(nanc, latent_dim))
        anc = gan.generator.predict(noise)

        # save model
        gan.generator.save_weights('save/models/generator_%sdata_%sepochs_%sanc.h5' % (len(X_train), epochs, nanc) )
    
    elif anc_type == 'gan':

        noise = np.random.normal(0, 1, size=(nanc, latent_dim))
        generator = GAN(X_train.shape[1], hidden_dim, latent_dim).generator
        generator.load_weights('save/models/generator_1200data_2000epochs_500anc.h5')
        # generator.load_weights('save/models/generator_%sdata_%sepochs_%sanc.h5' % (len(X_train), epochs, nanc) )

        anc = generator.predict(noise)
    
    elif anc_type == 'raw':
        
        idx = np.random.choice(len(X_train), nanc, replace=False)
        anc = X_train[idx]
    
    else:
        raise Exception('Unrecognized type')
    
    return anc

def sampling_iid(X_train, num_users, ndat):
    '''Sample I.I.D user data from dataset
    Return
    --------
    list of user index(key) and ndat samples index(value) 
    '''

    list_users, all_idx = [], [i for i in range(len(X_train))]
    for i in range(num_users):
        user_idx = set(np.random.choice(all_idx, size=ndat, replace=False))
        all_idx = list(set(all_idx) - user_idx) # remained index
        list_users.append(list(user_idx))
    
    return list_users

def sampling_noniid(X_train, label_train,  num_users, ndat, nlabel):
    '''Sample non-I.I.D user data from MNIST dataset
    Parameters
    ----------
    nlabel: int(1~9)
        number of different types of labels which each user has
    Return
    -------
    list of ndat samples index of each user
    '''
    num_shards, num_imgs = nlabel*num_users, int(ndat//nlabel)
    idx_shard = [i for i in range(num_shards)]
    
    ntrain = num_users * ndat

    # sort labels
    idx_label_pair = [(i, label) for i, label in enumerate(label_train[:ntrain])]
    idx_label_pair = sorted(idx_label_pair, key=lambda x:x[1])
    idx_sorted = [pair[0] for pair in idx_label_pair]

    # divide and assign nlabel number of label per user
    # dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    list_users = []
    for i in range(num_users):
        user_idx = []
        rand_set = set(np.random.choice(idx_shard, nlabel, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            user_idx = np.hstack((user_idx, idx_sorted[rand*num_imgs:(rand+1)*num_imgs]))
            user_idx = user_idx.astype(int).tolist() # convert dtype from float to int
        list_users.append(user_idx)

    return list_users


class GAN():
    def __init__(self, input_shape, hidden_dim=256, latent_dim=100, optimizer=Adam(0.0002, 0.5)):
        self.input_shape = (input_shape,)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # define generator instance
        self.generator = self.build_generator()
        
        
        # defince discriminator instance
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        # defined combined network
        self.combined = self.build_combined()
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    
    # generator model
    def build_generator(self):
        
        model = Sequential()
        
        model.add(Dense(self.hidden_dim, input_shape=(self.latent_dim,))) # 100 -> 256
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.hidden_dim*2)) # 256 -> 512
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.hidden_dim*4)) # 512 -> 1024
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
#         model.add(Reshape(self.input_shape))
        
        # model.summary()
        
        return model
    
    # discriminator model
    def build_discriminator(self):
        
        model = Sequential()
        
#         model.add(Flatten(input_shape=self.input_shape))
#         AttributeError: 'tuple' object has no attribute 'lower'
        model.add(Dense(self.hidden_dim*2, input_shape=self.input_shape)) # 256 -> 512
        model.add(LeakyReLU(0.2))
        model.add(Dense(self.hidden_dim)) # 512 -> 256
        model.add(LeakyReLU(0.2))
        model.add(Dense(1, activation='sigmoid')) # 256 -> number of class (0 or 1)
        
        # model.summary()
        
        return model
    
    # generatorとdiscriminatorを結合させる
    
    def build_combined(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model
        
    
    def train(self, X_train, epochs, batch_size=32):
        
        half_batch = int(batch_size / 2)
        
        num_batches = int(X_train.shape[0] / half_batch)
        print('Number of batches:', num_batches)
        
        
        for epoch in range(epochs+1):
            
            for iteration in range(num_batches):
            
                # ------ discriminator --------

                # pick up half-batch size of datafrom generator: G(x)
                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                gen_data = self.generator.predict(noise)

                # pick up half-batch size of dataset from dataset: D(x)
                idx = np.random.choice(X_train.shape[0], half_batch)
                true_data = X_train[idx]

                # make label for training: gen_data=0, true_data=1
                y_gen = np.zeros((half_batch, 1))
                y_true = np.ones((half_batch, 1))

                # learn discriminator
                d_loss_fake = self.discriminator.train_on_batch(gen_data, y_gen)
                d_loss_real = self.discriminator.train_on_batch(true_data, y_true)

                # average each loss
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                
                # ------- generator ---------

                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                
                # make label
                valid_y = np.ones((half_batch, 1))
                # valid_y = np.array([1] * half_batch)

                # train generator
                g_loss = self.combined.train_on_batch(noise, valid_y)
                
            # progress
            print('Epoch: %d, [D loss: %f] [G loss: %f]' % (epoch, d_loss, g_loss))

            if epoch % 200 == 0:
                self.save_fig(X_train, epoch)

    # generate 25 figures during training
    def save_fig(self, X_train, epoch):
        noise = np.random.normal(0, 1, size=(25, self.latent_dim))
        gen_img = self.generator.predict(noise)

        fig, axs = plt.subplots(5,5, figsize=(5,5))

        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(gen_img[i*5+j].reshape(28, 28), cmap='gray')
        
        fig.savefig('save/figures/%sdata_%sepoch.png' % (len(X_train), epoch))
        plt.close()