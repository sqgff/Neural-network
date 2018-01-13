import numpy as np
import tensorflow as tf
import rbm
from utils import batch_generator, gen_batches
import os


class Autoencoder_RBM(object):

    def __init__(self, name='autoencoder_rbm',
            rbm_layers=[10, 10],
            rbm_gauss_visible = False,
            rbm_stddev = 0.01,
            rbm_num_epochs=[10],
            rbm_gibbs_k=[1],
            rbm_batch_size=[10],
            rbm_learning_rate=[0.01],
            finetune_dropout=1,
            finetune_loss_func="cross_entropy",
            finetune_act_func=tf.nn.sigmoid,
            finetune_opt="sgd",
            finetune_learning_rate=0.01,
            finetune_num_epochs=10,
            finetune_batch_size=10,
            early_stopping_gap=1e-6,
            momentum=0.05,
            do_pretrain=False,
            tied_weights=False,
            save_path=None):
        """
        DBN constructor. Define the basic parameters for the model
        :param rbm_layers: a list containing number of hidden units, one element per hidden layer
        :param rbm_num_epochs: a list containing number of iterations, one element per layer
        :param rbm_gibbs_k: a list containing number of gibbs sampling, one element per layer
        :param rbm_batch_size: a list containing batch size, one element per layer
        :param rbm_learning_rate: a list containing learning rate, one element per layer
        :param finetune_dropout: dropout parameter
        :param finetune_loss_func: loss function for the reconstruction phase
        :param finetune_act_func: activation function for the finetuning phase
        :param finetune_opt: optimization method for the finetuning phase
        :param finetune_learning_rate: learning rate for the finetuning phase
        :param finetune_num_epochs: number of iterations for the finetuning phase
        :param finetune_batch_size: batch size for training the DBN model
        :param momentum: the momentum parameter
        """
        # Model configurations, to be specified by users
        self.tf_graph = tf.Graph()
        self.sess = None
        self.name = name
        self.model_path = save_path
        if save_path is None:
            self.model_path = os.getcwd() + '/' + self.name + '_model_saved'
        self.finetune_dropout = finetune_dropout
        self.finetune_loss_func = finetune_loss_func
        self.finetune_act_func = finetune_act_func
        self.finetune_opt = finetune_opt
        self.finetune_learning_rate = finetune_learning_rate
        self.finetune_num_epochs = finetune_num_epochs
        self.finetune_batch_size = finetune_batch_size
        self.early_stopping_gap=early_stopping_gap
        self.momentum = momentum
        self.stddev = rbm_stddev
        self.rbm_layers = rbm_layers
        self.gauss_visible = rbm_gauss_visible
        self.do_pretrain = do_pretrain
        self.tied_weights = tied_weights

        # Numerical variables, determined by the dataset
        self.n_features = None
        self.loss_summary = []

        # Tensforflow placeholders, for input data
        self.input_data = None
        self.keep_prob = None

        # Tensorflow variables
        self.encoding_w_ = None
        self.encoding_b_ = None
        self.decoding_w_ = None
        self.decoding_b_ = None

        # Tensorflow operations
        self.compression = None # Hidden layer that links encoding and decoding phase
        self.layer_nodes = None
        self.reconstruction = None
        self.reconstruction_loss = None
        self.train_step = None
        self.saver = None

        rbm_params = {
                'num_epochs': rbm_num_epochs,
                'k': rbm_gibbs_k,
                'batch_size': rbm_batch_size,
                'learning_rate': rbm_learning_rate}
        for param in rbm_params:
            if len(rbm_params[param]) != len(self.rbm_layers):
                rbm_params[param] = [rbm_params[param][0] for _ in self.rbm_layers]

        self.rbms = []
        self.rbm_graphs = []

        for l, layer in enumerate(self.rbm_layers):
            rbm_str = 'rbm_' + str(l+1)

            if l == 0 and rbm_gauss_visible:
                self.rbms.append(rbm.RBM(
                    name=self.name + '_' + rbm_str,
                    visible_unit_type='gauss',
                    gauss_stddev=rbm_stddev,
                    n_hidden=layer,
                    batch_size=rbm_params['batch_size'][l],
                    n_epochs=rbm_params['num_epochs'][l],
                    learning_rate=rbm_params['learning_rate'][l],
                    k=rbm_params['k'][l]))
            else:
                self.rbms.append(rbm.RBM(
                    name=self.name + '_' + rbm_str,
                    n_hidden=layer,
                    batch_size=rbm_params['batch_size'][l],
                    n_epochs=rbm_params['num_epochs'][l],
                    learning_rate=rbm_params['learning_rate'][l],
                    k=rbm_params['k'][l]))

            self.rbm_graphs.append(tf.Graph())

    def _create_placeholders(self, n_features):
        """
        create tensorflow placeholders for the models
        """
        self.input_data = tf.placeholder(tf.float32,
            [None, n_features],
            name='x-input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_probs')

    def _create_variables(self, train_set, n_features):
        """
        create tensorflow variables for the variable
        """
        if self.do_pretrain:
            self._create_encoding_variables_pretrain(train_set)
        else:
            self._create_encoding_variables_no_pretrain(n_features)
        self._create_decoding_variables()

    def _create_decoding_variables(self):
        """
        This function creates the tensorflow variables for the decoding phase
        """
        self.decoding_w_ = []
        self.decoding_b_ = []
        for l, layer in reversed(list(enumerate(self.rbm_layers))):
            dec_w = None
            dec_b = None
            if self.tied_weights:
                dec_w = tf.transpose(self.encoding_w_[l])
            else:
                dec_w = tf.Variable(tf.transpose(
                    self.encoding_w_[l].initialized_value()))
            dec_b = tf.Variable(tf.constant(0.1,
                shape=[dec_w.get_shape().dims[1].value]))
            self.decoding_w_.append(dec_w)
            self.decoding_b_.append(dec_b)

    def _create_encoding_variables_no_pretrain(self, n_features):
        """
        create tensorflow variables without pretraining
        """
        self.encoding_w_ = []
        self.encoding_b_ = []
        for l, layer in enumerate(self.rbm_layers):
            if l == 0:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[n_features, self.rbm_layers[l]],
                    stddev=0.1)))
            else:
                self.encoding_w_.append(tf.Variable(tf.truncated_normal(
                    shape=[self.rbm_layers[l-1], self.rbm_layers[l]],
                    stddev=0.1)))

            self.encoding_b_.append(tf.Variable(tf.constant(0.1, shape=[self.rbm_layers[l]])))


    def _create_encoding_variables_pretrain(self, train_set):
        """
        create tensorflow variables with pretraining
        Suppose that the rbm layers have been pretrained
        """
        def sigmoid(x):
            return 1 / (1.0 + np.exp(-x))
        def compute_next_train(train_set, w, b):
            probs = sigmoid(np.dot(train_set, w) + b)
            return np.maximum(np.sign(probs - np.random.rand(probs.shape[0],
                probs.shape[1])),0)

        self.encoding_w_ = []
        self.encoding_b_ = []
        next_train = train_set
        for l, rbm in enumerate(self.rbms):
            rbm.fit(next_train)
            #params_tmp = rbm.get_parameters(self.rbm_graphs[l])
            params_tmp = rbm.trained_params
            #self.encoding_w_.append(tf.Variable(params_tmp['W'][0]))
            self.encoding_w_.append(tf.Variable(params_tmp['W']))
            #self.encoding_b_.append(tf.Variable(params_tmp['hbias'][0]))
            self.encoding_b_.append(tf.Variable(params_tmp['hbias']))
            #next_train = compute_next_train(next_train, params_tmp['W'][0],
                    #params_tmp['hbias'][0])
            next_train = compute_next_train(next_train, params_tmp['W'],
                    params_tmp['hbias'])
            print('hidden layer %d has been pretrained' % (l+1))

    def _create_encoding_layers(self):
        """
        Create the encoding layers for the autoencoder model
        return: output of the final encoding layer
        """
        next_train = self.input_data
        self.layer_nodes = []

        for l, layer in enumerate(self.rbm_layers):
            y_act = tf.add(tf.matmul(next_train, self.encoding_w_[l]),
                    self.encoding_b_[l])

            layer_y = None
            if self.finetune_act_func is not None:
                layer_y = self.finetune_act_func(y_act)
            next_train = tf.nn.dropout(layer_y, self.keep_prob)
            self.layer_nodes.append(next_train)

        return next_train

    def _create_decoding_layers(self):
        """
        Create the decoding pahse for the autoencoder model
        return: output of the final decoding layer, which should be the reconstruction of the input data
        """
        next_train = self.compression

        for l, layer in enumerate(self.rbm_layers):
            y_act = tf.add(tf.matmul(next_train, self.decoding_w_[l]),
                    self.decoding_b_[l])

            layer_y = None
            if l == len(self.rbm_layers) - 1 and self.gauss_visible:
                layer_y = tf.truncated_normal(shape=[1, self.decoding_b_[l].get_shape().dims[0].value], mean=y_act, stddev = self.stddev)
            elif self.finetune_act_func is not None:
                layer_y = self.finetune_act_func(y_act)
            next_train = tf.nn.dropout(layer_y, self.keep_prob)
            self.layer_nodes.append(next_train)

        return next_train

    def compute_reconstructed_error(self, score, func):
        """
        compute the loss of trained model
        """
        if func == 'cross_entropy':
            clip_inf = tf.clip_by_value(score, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(1 - score, 1e-10, float('inf'))
            return - tf.reduce_mean(tf.add(
                tf.multiply(self.input_data, tf.log(clip_inf)),
                tf.multiply(1 - self.input_data, tf.log(clip_sup))))
        elif func == 'mse':
            return tf.sqrt(
                    tf.reduce_mean(
                        tf.square(tf.subtract(self.input_data, score)),1))
        else:
            return None

    def _build_model(self, n_features, train_set):
        """
        Build the model, creating the computational graph

        The graph is created for the finetuning phase, i.e.  after unsupervisied pretraining
        """
        self._create_placeholders(n_features)
        self._create_variables(train_set, n_features)
        self.compression = self._create_encoding_layers()
        self.reconstruction = self._create_decoding_layers()

        self.reconstruction_loss = self.compute_loss(self.reconstruction,
                self.finetune_loss_func)
        self.reconstructed_error = self.compute_reconstructed_error(self.reconstruction,
                                self.finetune_loss_func)
        self.train_step = self.trainer(self.reconstruction_loss)
        self.saver = tf.train.Saver()

    def trainer(self, loss):
        """
        Define the train step, using the optimization method required.
        """
        if self.finetune_opt == 'sgd':
            return tf.train.GradientDescentOptimizer(self.finetune_learning_rate).minimize(loss)
        elif self.finetune_opt == 'adagrad':
            return tf.train.AdagradOptimizer(self.finetune_learning_rate).minimize(loss)

    def compute_loss(self, score, func):
        """
        compute the loss of trained model
        """
        if func == 'cross_entropy':
            clip_inf = tf.clip_by_value(score, 1e-10, float('inf'))
            clip_sup = tf.clip_by_value(1 - score, 1e-10, float('inf'))
            return - tf.reduce_mean(tf.add(
                tf.multiply(self.input_data, tf.log(clip_inf)),
                tf.multiply(1 - self.input_data, tf.log(clip_sup))))
        elif func == 'mse':
            return tf.sqrt(
                    tf.reduce_mean(
                        tf.square(tf.subtract(self.input_data, score))))
        else:
            return None


    def fit(self, train_set, validation_set=None, test_set=None, graph=None):
        """
        Fit the dbn model. Perform the pretrain step if required.
        """
        self.n_features = train_set.shape[1]
        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            self._build_model(self.n_features, train_set)
            with tf.Session(graph=g) as self.sess:
                tf.global_variables_initializer().run()
                self._train_model(train_set, validation_set)

                if test_set is not None:
                    print("Begin reconstruction phase on test dataset:")
                    feed_test = {
                            self.input_data: test_set,
                            self.keep_prob: 1}
                    actions = [self.compression, self.reconstruction]
                    compres, recons = self.sess.run(actions, feed_dict=feed_test)
                    return compres, recons, self.loss_summary


    def _train_model(self, train_X, validation_X):
        """
        This function perform the training process
        """
        feed_train = {
                self.input_data: train_X,
                self.keep_prob: self.finetune_dropout}
        feed_validation = {
                self.input_data: validation_X,
                self.keep_prob: 1}

        shuff = train_X
        old_iter, old_loss = 0, 10000
        for iteration in range(1, self.finetune_num_epochs+1):
            np.random.shuffle(shuff)
            batches = [_ for _ in gen_batches(shuff, self.finetune_batch_size)]

            for batch in batches:
                feed_batch = {
                        self.input_data: batch,
                        self.keep_prob: self.finetune_dropout}
                self.sess.run(self.train_step, feed_dict=feed_batch)

            print('Iter %d' % iteration, end=': ')
            loss_train = self.sess.run(self.reconstruction_loss,
                    feed_dict=feed_train)
            if validation_X is not None:
                loss_validation = self.sess.run(self.reconstruction_loss,
                        feed_dict=feed_validation)
                reconstructed_error = self.sess.run(self.reconstructed_error,
                                                feed_dict=feed_validation)
                self.loss_summary.append(loss_validation)
                print('Training set: current loss %f' % loss_train, end='  ||  ')
                print('Validation set: current loss %f' % loss_validation)
                print('Validation: ', np.argsort(-reconstructed_error)[:25] )
                if loss_validation > old_loss and (iteration - old_iter) > 5:
                    return
                if loss_validation < old_loss:
                    old_loss = loss_validation
                    old_iter = iteration
            else:
                print('Training set: current loss %f' % loss_train)












