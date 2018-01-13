"""
An implementation of restricted boltzmann machine based on tensorflow

initially for the project 3A
"""
import numpy as np
import tensorflow as tf
import os
from utils import batch_generator

class RBM(object):
    def __init__(self, name="rbm",
            visible_unit_type='binary',
            n_hidden=500,
            batch_size=64,
            k=1,
            n_epochs=10,
            learning_rate=0.5,
            save_path=None):
        """
        RBM constrcutor. Define the basic parameters of the model
        """
        self.tf_graph = tf.Graph()
        self.sess = None
        self.name = name
        self.model_path = save_path
        if save_path is None:
            self.model_path = os.getcwd() + "/" + self.name + "_model_saved"
        #if not os.path.exists(self.model_path):
        #    os.makedirs(self.model_path)
        #self.model_path = self.model_path + self.name

        self.visible_unit_type = visible_unit_type
        self.n_visible = None
        self.n_hidden = n_hidden

        self.visible_units_placeholder = None
        self.chain_end = None
        self.W = None
        self.vbias = None
        self.hbias = None

        self.validation_set_placeholder = None
        self.batch_size = batch_size
        self.k = k
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.params = [self.W, self.vbias, self.hbias]
        self.saver = None

        self.loss = None
        self.reconstruction_error = None
        self.update_W = None
        self.update_vbias = None
        self.update_hbias = None

    def _create_variables(self, n_visible):
        """create tensorflow variables for the model"""

        abs_val = 4 * np.sqrt(6. / (n_visible + self.n_hidden))
        self.W = tf.Variable(tf.random_uniform(shape=[n_visible, self.n_hidden], minval=-abs_val, maxval=abs_val))

        self.vbias = tf.Variable(tf.zeros(shape=[n_visible], dtype=tf.float32))

        self.hbias = tf.Variable(tf.zeros(shape=[self.n_hidden], dtype=tf.float32))


    def prob_to_sample(self, probs):
        """This function takes a vector of mean and implement binomial sampling using the uniform distribution"""
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def sample_h_given_v(self, v0_sample):
        """
        This function infers states of hidden units, given visible units.
        """
        # We should use the function propup to infer states of hidden units
        hidden_activation = tf.add(tf.matmul(
            tf.reshape(v0_sample, [-1, self.n_visible]), self.W), self.hbias)
        h_mean = tf.nn.sigmoid(hidden_activation)
        h_sample = self.prob_to_sample(h_mean)
        return [hidden_activation, h_mean, h_sample]

    def sample_v_given_h(self, h0_sample):
        """
        This function infers states of visible units, given hidden units
        """
        # we should use the function propdown to infer states of visible units
        visible_activation = tf.add(tf.matmul(h0_sample, tf.transpose(self.W)),
                self.vbias)
        if self.visible_unit_type == 'binary':
            return tf.nn.sigmoid(visible_activation)
        elif self.visible_unit_type == 'gauss':
            return tf.truncated_normal((1, self.n_visible),
                    mean=visible_activation, stddev=0.01)

    def gibbs_hvh(self, h0_sample):
        v1_probs = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_probs)
        return [v1_probs, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        v1_probs = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, v1_probs]

    def free_energy(self, v_sample):
        """Function to compute the free energy"""
        flat_v = tf.reshape(v_sample, [-1, self.n_visible])
        wx_b = tf.matmul(flat_v, self.W) + self.hbias
        vbias_term = tf.matmul(flat_v, tf.reshape(self.vbias, [-1, 1]))
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(wx_b)))
        return -hidden_term - vbias_term

    def _build_model(self):
        self.visible_units_placeholder = tf.placeholder(tf.float32, shape=[None, self.n_visible])
        self._create_variables(self.n_visible)

        pre_sigmoid, h_mean, h_sample = self.sample_h_given_v(self.visible_units_placeholder)

        positive = None
        if self.visible_unit_type == 'binary':
            positive = tf.matmul(tf.transpose(self.visible_units_placeholder),
                    h_sample)
        elif self.visible_unit_type == 'gauss':
            positive = tf.matmul(tf.transpose(self.visible_units_placeholder),
                    h_mean)

        # k gibbs sampling
        nh_sample = h_sample
        for i in range(self.k):
            vk_probs, pre_sigmoid_hk, hk_mean, nh_sample = self.gibbs_hvh(nh_sample)

        self.chain_end = vk_probs

        negative = tf.matmul(tf.transpose(self.chain_end), nh_sample)
        # CD-k Loss
        self.loss = tf.reduce_mean(self.free_energy(self.visible_units_placeholder)) - tf.reduce_mean(self.chain_end)

        self.update_W = tf.assign_add(self.W, self.learning_rate * (positive - negative) / self.batch_size)
        self.update_hbias = tf.assign_add(self.hbias, tf.multiply(self.learning_rate, tf.reduce_mean(tf.subtract(h_sample, nh_sample), 0)))
        self.update_vbias = tf.assign_add(self.vbias, tf.multiply(self.learning_rate, tf.reduce_mean(tf.subtract(self.visible_units_placeholder, self.chain_end), 0)))

        self.validation_set_placeholder = tf.placeholder(tf.float32, shape=[None, self.n_visible])
        self.reconstruction_error = self.compute_reconstruction_error(self.validation_set_placeholder)

        self.saver = tf.train.Saver()

    def compute_reconstruction_error(self, validation_set):
        """This function computes the reconstruction error, using the trained parameters and validation set"""
        _, _, _, v_probs = self.gibbs_vhv(validation_set)
        err = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(validation_set,
            v_probs))))
        return err

    def fit(self, train_set, validation_set=None, graph=None):
        """fit a mode given data"""
        self.n_visible = train_set.shape[1]

        g = graph if graph is not None else self.tf_graph
        with g.as_default():
            self._build_model()
            #optimization_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        #self.sess = tf.Session(graph=self.graph)
        with tf.Session(graph=g) as sess:

            self.sess = sess
            tf.global_variables_initializer().run()
            for iteration in range(1, self.n_epochs + 1):
                idx = np.random.permutation(len(train_set))
                data = train_set[idx]

                # For debug
                #print("W: ", self.W)
                #print("hbias: ", self.hbias)
                #print("vbias: ", self.vbias)
                #print("data shape: ", data.shape)


                for batch in batch_generator(self.batch_size, data):
                    #if len(batch) < self.batch_size:
                        # pad with zeros
                        #pad = np.zeros((self.batch_size - batch.shape[0], batch.shape[1]), dtype=batch.dtype)
                        #bacth = np.vstack((batch, pad))
                    #_, current_loss = sess.run([optimization_step, self.loss], feed_dict={self.visible_units_placeholder: batch})
                    updates = [self.update_W, self.update_hbias, self.update_vbias]
                    # debug
                    updates.append(self.loss)
                    feed_inner = {self.visible_units_placeholder: batch}
                    _, _, _, current_loss = sess.run(updates, feed_dict=feed_inner)
                    #print("current_loss: ", current_loss)

                if validation_set is not None:
                    feed_outer = {self.validation_set_placeholder: validation_set}
                    loss_recons = sess.run(self.reconstruction_error, feed_dict=feed_outer)
                    print("iter ", iteration, ": ", loss_recons)
            tf.add_to_collection("W", self.W)
            tf.add_to_collection("hbias", self.hbias)
            tf.add_to_collection("vbias", self.vbias)
            self.saver.save(sess, self.model_path)


    def get_parameters(self, graph=None):
        g = graph if graph is not None else self.tf_graph
        with g.as_default():
            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph((self.model_path + ".meta"))
                new_saver.restore(sess, tf.train.latest_checkpoint("./"))
                #print("W", sess.run(tf.get_collection("W")))
                return {
                        "W": sess.run(tf.get_collection("W")),
                        "hbias": sess.run(tf.get_collection("hbias")),
                        "vbias": sess.run(tf.get_collection("vbias"))}







