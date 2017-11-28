# """ Simple convolutional neural network.
# UNAM IIMAS
# AUthor: Ivette Velez
# Tutor:  Caleb Rascon
# Co-tutor: Gibran Fuentes
# To run the model: python cnn6_V1_VGG16.py --learning_rate 0.002 --num_epochs 20 --train_dir dataset --batch_size 100
# """

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train and Eval the MNIST network.
This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files
for context.
YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import os
import sys
import time
import numpy as np
import random
import soundfile as sf
import tensorflow as tf
from scipy import ndimage
from scipy import signal
import sklearn.preprocessing as pps
from sklearn.preprocessing import StandardScaler

#from tensorflow.examples.tutorials.mnist import mnist
from tensorflow.contrib import layers

# silences Tensorflow boot logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Using just one GPU in case of GPU 
os.environ['CUDA_VISIBLE_DEVICES']= '0'


# Basic model parameters as external flags.
FLAGS = None

FILE_TRAIN = 'train_speakers.txt'
FILE_VALID = 'valid_speakers.txt'

FILE_TRAIN_TAG = 'train_speakers_tag.txt'
FILE_VALID_TAG = 'valid_speakers_tag.txt'

# Constants on the model
WINDOW = 4*16000 
PART = 2
MS = 1.0/16000
NPERSEG = int(0.025/MS)
NOVERLAP = int(0.015/MS)
NFFT =1024

SIZE_FFT = int(NFFT/4)
SIZE_COLS = int(np.ceil((WINDOW - NPERSEG)/(NPERSEG - NOVERLAP)))

print(SIZE_COLS)

TOTAL_DATA_TRAIN = 4378
TOTAL_DATA_VALID = 1759
VAD = 0.05

L_LABEL = 7
IN_HEIGHT = SIZE_FFT
IN_WIDTH = SIZE_COLS
CHANNELS = 1
POOL_LAYERS = 4

WIDTH_AFTER_CONV = int(np.ceil(float(IN_WIDTH)/float(2**POOL_LAYERS)))
HEIGHT_AFTER_CONV = int(np.ceil(float(IN_HEIGHT)/float(2**POOL_LAYERS)))

print(WIDTH_AFTER_CONV)
print(HEIGHT_AFTER_CONV)

# Function that splits and string to get the desire part
def get_part(part,string):
  aux = string.split('/')
  a = aux[len(aux)-part-1]
  return a

# function that gets the numeric part of a string
def get_int(string):
  return int(''.join([d for d in string if d.isdigit()]))

class cnn7:

  def __init__(self):
    """ Creates the model """
    self.def_input()
    self.def_variable()
    self.def_params()
    self.def_model()
    self.def_output()
    self.def_loss()
    self.def_metrics()
    self.add_summaries()

  def conv2d(self, x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(self, x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

  def weight_variable(self,shape):
    initializer = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer)
    
  def bias_variable(self,shape):
    initializer = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initializer)

  def def_input(self):
    """ Defines inputs """
    with tf.name_scope('input'):

      # Defining the entrance of the model
      self.X1 = tf.placeholder(tf.float32, [None, IN_HEIGHT, IN_WIDTH, CHANNELS], name='X1')
      self.Y = tf.placeholder(tf.float32, [None, L_LABEL], name='Y')

      self.g_step = tf.contrib.framework.get_or_create_global_step()

  def def_variable(self):
    self.size_batch = FLAGS.batch_size

  def def_params(self):
    """ Defines model parameters """
    with tf.name_scope('params'):

      # First convolutional layer
      with tf.name_scope('conv1'):
        self.W_cn1 = self.weight_variable([3, 3, 1, 32])
        self.b_cn1 = self.bias_variable([32])

      # Second convolutional layer
      with tf.name_scope('conv2'):
        self.W_cn2 = self.weight_variable([3, 3, 32, 64])
        self.b_cn2 = self.bias_variable([64])

      # Third convolutional layer
      with tf.name_scope('conv3'):
        self.W_cn3 = self.weight_variable([3, 3, 64, 128])
        self.b_cn3 = self.bias_variable([128])

      # Fourth Convolutional layer
      with tf.name_scope('conv4'):
        self.W_cn4 = self.weight_variable([3, 3, 128, 256])
        self.b_cn4 = self.bias_variable([256])

      # First fully connected layer      
      with tf.name_scope('fc1'):
        self.W_fc1 = self.weight_variable([HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV * 256, 1024])
        self.b_fc1 = self.bias_variable([1024])

      # Second fully connected layer      
      with tf.name_scope('fc2'):
        self.W_fc2 = self.weight_variable([1024, 1024])
        self.b_fc2 = self.bias_variable([1024])

      # Third fully connected layer
      with tf.name_scope('fc3'):
        self.W_fc3 = self.weight_variable([1024, L_LABEL])
        self.b_fc3 = self.bias_variable([L_LABEL])        


  def def_model(self):
    """ Defines the model """
    W_cn1 = self.W_cn1
    b_cn1 = self.b_cn1
    W_cn2 = self.W_cn2
    b_cn2 = self.b_cn2
    W_cn3 = self.W_cn3
    b_cn3 = self.b_cn3
    W_cn4 = self.W_cn4
    b_cn4 = self.b_cn4
    W_fc1 = self.W_fc1
    b_fc1 = self.b_fc1
    W_fc2 = self.W_fc2
    b_fc2 = self.b_fc2
    W_fc3 = self.W_fc3
    b_fc3 = self.b_fc3
  
    # First convolutional layers for the first signal
    with tf.name_scope('conv1a'):
      h_cn1a = tf.nn.relu(self.conv2d(self.X1, W_cn1) + b_cn1)

    # First pooling layer for the first signal
    with tf.name_scope('pool1a'):
      #h_pool1a = self.max_pool_2x1(h_cn2a)
      h_pool1a = self.max_pool_2x2(h_cn1a)    

    # Second convolutional layers for the first signal
    with tf.name_scope('conv2a'):
      h_cn2a = tf.nn.relu(self.conv2d(h_pool1a, W_cn2) + b_cn2)

    # First pooling layer for the first signal
    with tf.name_scope('pool2a'):
      #h_pool1a = self.max_pool_2x1(h_cn2a)
      h_pool2a = self.max_pool_2x2(h_cn2a)

    # Third convolutional layers for the first signal
    with tf.name_scope('conv3a'):
      h_cn3a = tf.nn.relu(self.conv2d(h_pool2a, W_cn3) + b_cn3)

    # First pooling layer for the first signal
    with tf.name_scope('pool3a'):
      #h_pool1a = self.max_pool_2x1(h_cn2a)
      h_pool3a = self.max_pool_2x2(h_cn3a)    

    # Fourth convolutional layers for the first signal
    with tf.name_scope('conv4a'):
      h_cn4a = tf.nn.relu(self.conv2d(h_pool3a, W_cn4) + b_cn4)

    # Second pooling layer for the first signal
    with tf.name_scope('pool4a'):
      h_pool4a = self.max_pool_2x2(h_cn4a)

    # First fully connected layer
    with tf.name_scope('fc1'):
      h_concat1_flat = tf.reshape(h_pool4a, [-1, HEIGHT_AFTER_CONV * WIDTH_AFTER_CONV *256]) 
      h_mat =  tf.matmul(h_concat1_flat, W_fc1)
      h_fc1 = tf.nn.relu(h_mat + b_fc1)

    # Second fully connected layer
    with tf.name_scope('fc2'):
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    # Third fully connected layer
    with tf.name_scope('fc3'):
      self.Y_logt = tf.matmul(h_fc2, W_fc3) + b_fc3
      self.Y_sig = tf.nn.sigmoid(self.Y_logt, name='Y_sig')
      self.Y_pred = tf.round(self.Y_sig, name='Y_pred')

  def def_output(self):
    """ Defines model output """
    with tf.name_scope('output'):
      self.label_pred = tf.argmax(self.Y_pred, 1, name='label_pred')
      self.label_true = tf.argmax(self.Y, 1, name='label_true')

  def def_loss(self):
    """ Defines loss function """
    with tf.name_scope('loss'):

      #cross entropy
      self.cross_entropy =  tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=self.Y_logt)*10
      self.loss = tf.reduce_mean(self.cross_entropy)
      

  def def_metrics(self):
    """ Adds metrics """
    with tf.name_scope('metrics'):
      self.cmp_labels = tf.equal(self.Y, self.Y_pred)
      self.accuracy = tf.reduce_sum(tf.cast(self.cmp_labels, tf.float32), name='accuracy')
      self.acc_batch = (self.accuracy/(self.size_batch*L_LABEL))*100

  def add_summaries(self):
    """ Adds summaries for Tensorboard """
    # defines a namespace for the summaries
    with tf.name_scope('summaries'):
      # adds a plot for the loss
      tf.summary.scalar('loss', self.loss)
      #tf.summary.scalar('accuracy', self.accuracy)
      tf.summary.scalar('accuracy', self.acc_batch)
      # groups summaries
      self.summary = tf.summary.merge_all()

  def train(self):

    # Creating a folder where to save the parameter
    file_path = str(sys.argv[0]) +'_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs)
    os.mkdir(file_path)

    # Creating a file to write the loss and acurracy
    output_file = open(file_path+'_results.txt', 'w')

    """ Trains the model """
    # creates optimizer
    grad = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    
    # # setup minimize function
    optimizer = grad.minimize(self.loss)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    # opens session
    with tf.Session() as sess:
      
      # writers for TensorBorad
      train_writer = tf.summary.FileWriter('graphs/train_XS_XA_GRAD_4S_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      valid_writer = tf.summary.FileWriter('graphs/valid_XS_XA_GRAD_4S_' + str(FLAGS.learning_rate)+'_'+str(FLAGS.num_epochs))
      train_writer.add_graph(sess.graph)

      # initialize variables (params)
      sess.run(init_op)

      # Initializing the step for train and validation
      step_train = 1
      step_valid = 1
      acc_train = 0
      acc_valid = 0
      
      # Compute for the desired number of epochs.
      for n_epochs in range(FLAGS.num_epochs):

        k_limit = 2

        for step_file in range(0,k_limit):

          complete_LA = []
          complete_y_sig = []
          commplete_y_pred = []
          complete_y_real = []

          if step_file == 0:
            total_data = TOTAL_DATA_TRAIN
            input_file  = open(FILE_TRAIN,'r')
            input_file_tag  = open(FILE_TRAIN_TAG,'r')

          elif step_file == 1:      
            total_data = TOTAL_DATA_VALID
            input_file  = open(FILE_VALID,'r')
            input_file_tag  = open(FILE_VALID_TAG,'r')

          # Determing the rows of the file to write
          #if total_data < ROWS_FILE:
          if total_data < FLAGS.batch_size:
            rows_tf = total_data
          else:
            rows_tf = FLAGS.batch_size

          # Moving audio routes to a matrix
          matrix = []
          for line in input_file:
            row = line.rstrip()
            matrix.append(row)

          # Moving audio routes to a matrix
          matrix_tag = []
          for line in input_file_tag:
            row = line.rstrip()
            matrix_tag.append(row)  

          data_per_row = int(np.ceil(float(total_data)/float(2*len(matrix))))
          rows = len(matrix)

          # Doing a permutation of the data        
          matrix = np.array(matrix)
          matrix_tag = np.array(matrix_tag)

          #data_permutation = np.random.permutation(len(matrix))
          data_permutation = np.random.permutation(matrix.shape[0])
          matrix = matrix[data_permutation]
          matrix_tag = matrix_tag[data_permutation]

          X1 = []
          Y = []
          list_audios = []

          i = 0
          total = 0
          j = 0

          while i < total_data:

            flag_model = False

            chosen_audio_1 = matrix[j]
            audio_1,samplerate = sf.read(chosen_audio_1)

            # getting the spectrogram
            f, t, Sxx1 = signal.spectrogram(audio_1, samplerate,  window=('hamming'), nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, detrend='constant', return_onesided=False, scaling='density', axis=-1)
            Hxx1 = StandardScaler().fit_transform(Sxx1)

            data_audio_1 = np.reshape(Hxx1[0:SIZE_FFT,:],(SIZE_FFT,Hxx1.shape[1],1))
            file_tag = open(matrix_tag[j],'r')

            # Getting the class
            matrix_aux = []
            for line in file_tag:
              row = line.rstrip().split(',')
              matrix_aux.append(row)
            
            y_aux = [0,0,0,0,0,0,0]

            if matrix_aux[9][1].count('c')>0:
              y_aux[0] = 1

            if matrix_aux[9][1].count('m')>0:
              y_aux[1] = 1

            if matrix_aux[9][1].count('f')>0:
              y_aux[2] = 1

            if matrix_aux[9][1].count('v')>0:
              y_aux[3] = 1

            if matrix_aux[9][1].count('p')>0:
              y_aux[4] = 1

            if matrix_aux[9][1].count('b')>0:
              y_aux[5] = 1

            if matrix_aux[9][1].count('o')>0:
              y_aux[6] = 1

            # Filling the matrixes with the data
            X1.append(data_audio_1)
            Y.append(y_aux)
            list_audios.append(chosen_audio_1)
            total+=1
            i+=1

            if total>= rows_tf:
              flag_model = True

            # If the file must be written
            if flag_model == True:

              X1_array = np.array(X1)
              Y_array = np.array(Y)
              list_audios_array = np.array(list_audios)

              permutation = np.random.permutation(X1_array.shape[0])
              X1_array = X1_array[permutation,:]
              Y_array = Y_array[permutation]
              list_audios_array = list_audios_array[permutation]              


              # Running the apropiate model
              # Train
              if step_file == 0:

                # evaluation with train data
                feed_dict = {self.X1: X1_array, self.Y : Y_array}
                fetches = [self.Y_pred, self.cmp_labels, optimizer, self.loss, self.accuracy, self.summary, self.W_cn1, self.b_cn1, self.W_cn2, self.b_cn2, self.W_cn3, self.b_cn3, self.W_cn4, self.b_cn4, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3]
                y_pred, cmp_labels,_,train_loss, train_acc, train_summary, W_cn1, b_cn1, W_cn2, b_cn2, W_cn3, b_cn3, W_cn4, b_cn4, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3 = sess.run(fetches, feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step_train)

                acc_train = acc_train + train_acc

                # Printing the results every 100 batch
                if step_train % 10 == 0:
                
                  msg = "I{:3d} loss_train: ({:6.8f}), acc_train(batch, global): ({:6.8f},{:6.8f})"
                  msg = msg.format(step_train, train_loss, train_acc/(FLAGS.batch_size*L_LABEL), acc_train/(FLAGS.batch_size*step_train*L_LABEL))
                  print(msg)
                  output_file.write(msg + '\n')

                step_train += 1

              # Validation
              elif step_file == 1:
                
                # evaluation with train data
                feed_dict = {self.X1: X1_array, self.Y : Y_array}
                fetches = [self.Y_pred, self.Y_sig, self.loss, self.accuracy, self.summary]
                y_pred, y_sig, valid_loss, valid_acc, valid_summary = sess.run(fetches, feed_dict=feed_dict)
                valid_writer.add_summary(valid_summary, step_train)

                # appending the data for y_pred and y_sig and y_real

                acc_valid = acc_valid + valid_acc

                if step_valid % 10 == 0:
                  msg = "I{:3d} loss_val: ({:6.8f}), acc_val(batch, global): ({:6.8f},{:6.8f})"
                  msg = msg.format(step_valid, valid_loss, valid_acc/(FLAGS.batch_size*L_LABEL), acc_valid/(FLAGS.batch_size*step_valid*L_LABEL))
                  print(msg)
                  output_file.write(msg + '\n')

                step_valid += 1

                if complete_LA == []:
                  complete_LA = list_audios_array
                  complete_y_sig = y_sig
                  commplete_y_pred = y_pred
                  complete_y_real = Y_array
                else:
                  complete_LA = np.concatenate((complete_LA,list_audios_array),axis = 0)
                  complete_y_sig = np.concatenate((complete_y_sig,y_sig),axis = 0)
                  commplete_y_pred = np.concatenate((commplete_y_pred, y_pred),axis = 0)
                  complete_y_real = np.concatenate((complete_y_real, Y_array),axis = 0)



              total = 0
              X1 = []
              Y = []
              list_audios = []
            
            j+=1
            

      # Saving the parameters of the model
      np.save(file_path+'/'+ str(n_epochs) + "_W_cn1",W_cn1)
      np.save(file_path+'/'+ str(n_epochs) + "_b_cn1",b_cn1)
      np.save(file_path+'/'+ str(n_epochs) + "_W_cn2",W_cn2)
      np.save(file_path+'/'+ str(n_epochs) + "_b_cn2",b_cn2)
      np.save(file_path+'/'+ str(n_epochs) + "_W_cn3",W_cn3)
      np.save(file_path+'/'+ str(n_epochs) + "_b_cn3",b_cn3)
      np.save(file_path+'/'+ str(n_epochs) + "_W_cn4",W_cn4)
      np.save(file_path+'/'+ str(n_epochs) + "_b_cn4",b_cn4)
      np.save(file_path+'/'+ str(n_epochs) + "_W_fc1",W_fc1)
      np.save(file_path+'/'+ str(n_epochs) + "_b_fc1",b_fc1)
      np.save(file_path+'/'+ str(n_epochs) + "_W_fc2",W_fc2)
      np.save(file_path+'/'+ str(n_epochs) + "_b_fc2",b_fc2)
      np.save(file_path+'/'+ str(n_epochs) + "_W_fc3",W_fc3)
      np.save(file_path+'/'+ str(n_epochs) + "_b_fc3",b_fc3)

      
      # Saving the neccessary data
      np.save(file_path+'/'+ "list_audios",complete_LA)
      np.save(file_path+'/'+ "y_sig",complete_y_sig)
      np.save(file_path+'/'+ "y_pred",commplete_y_pred)
      np.save(file_path+'/'+ "y_real",complete_y_real)


def run():

  # defines our model
  model = cnn7()

  # trains our model
  model.train()


def main(args):
  run()
  return 0


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.00001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=10,
      help='Number of epochs to run trainer.'
  )

  parser.add_argument(
      '--batch_size',
      type=int,
      default=50,
      help='Batch size.'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)