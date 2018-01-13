from rbm import RBM
from dbn import DeepBeliefNetwork

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# Download the data in the working directory
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

#training_data = {image: mnist.train.images, label: mnist.train.labels}
#validation_data = {image: mnist.validation.images, label: mnist.validation.labels}
#test_data = {image: mnist.test.images, label: mnist.test.labels}
train_dataset = mnist.train.images
train_labels = mnist.train.labels
validation_dataset = mnist.validation.images
validation_labels = mnist.validation.labels
test_dataset = mnist.test.images
test_labels = mnist.test.labels
print("The shape of the dataset for training: ", train_dataset.shape, train_labels.shape)
print("The shape of the dataset for validation: ", validation_dataset.shape, validation_labels.shape)
print("The shape of the dataset for test: ", test_dataset.shape, test_labels.shape)

# Test for RBM
#rbm_model = RBM(visible_unit_type='gauss', batch_size=100, n_epochs=10, learning_rate=0.01)
#rbm_model.fit(train_dataset, validation_set = validation_dataset)
#rbm_model.get_parameters(tf.Graph())

# Test for DBN
dbn_model = DeepBeliefNetwork(
        name='dbn',
        rbm_layers=[1024, 512],
        rbm_gauss_visible=True,
        finetune_num_epochs=50,
        do_pretrain=True)
dbn_model.fit(train_dataset, train_labels, validation_dataset, validation_labels)
