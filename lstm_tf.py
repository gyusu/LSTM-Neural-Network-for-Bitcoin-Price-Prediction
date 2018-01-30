# coding=utf-8
import numpy as np
import tensorflow as tf

class LSTM:

    def __init__(self, session: tf.Session, x_window_size: int, ncols: int,
               name: str="main") -> None:
        """
        Args:
          session (tf.Session): Tensorflow session
          x_window_size: sequence length
          ncols: number of columns (e.g. open,high,low,close -> 4)
          name (str, optional): TF Graph will be built under this name scope
        """

        self.session = session
        self.x_window_size = x_window_size
        self.ncols = ncols
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=150, l_rate=0.001) -> None:
        """
        # Input -> LSTM layer -> FC layer -> Output
        Args:
          h_size (int, optional): Hidden layer dimension
          l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):

            # [batch_size(None), window_size, num of cols]
            self._X = tf.placeholder(tf.float32, [None, self.x_window_size, self.ncols], name="input_x")

            # [batch_size(batch_size(None), output_dim]
            self._Y = tf.placeholder(tf.float32, [None, 1])

            # hidden units = 150
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=150, state_is_tuple=True)
            outputs, _states = tf.nn.dynamic_rnn(cell, self._X, dtype=tf.float32)

            # outputs[:, -1] 는 LSTM layer의 output 중에서 마지막 것만 사용한다는 의미.
            self._Y_pred = tf.contrib.layers.fully_connected(
                outputs[:, -1], 1, activation_fn=tf.nn.tanh)

            # cost/loss
            self._loss = tf.reduce_sum(tf.square(self._Y_pred - self._Y))

            # optimizer
            self._optimizer = tf.train.AdamOptimizer(l_rate)
            self._train = self._optimizer.minimize(self._loss)

            # RMSE
            self._targets = tf.placeholder(tf.float32, [None, 1])
            self._predictions = tf.placeholder(tf.float32, [None, 1])
            self._rmse = tf.sqrt(tf.reduce_mean(tf.square(self._targets - self._predictions)))

    def train(self, epochs, steps_per_epoch, data_gen_train):
        self.session.run(tf.global_variables_initializer())

        for i in range(epochs):
            for j in range(steps_per_epoch):
                batch_x, batch_y = next(data_gen_train)
                batch_y = np.array(batch_y).reshape(-1, 1)
                _, step_loss = self.session.run([self._train, self._loss], feed_dict={
                    self._X: batch_x, self._Y: batch_y})
                print("[epoch: {}, step: {}] loss: {}".format(i, j, step_loss))

    def predict(self, steps_test, data_gen_test, true_values):
        rmse_list = np.array([])
        test_predict_list = np.array([])

        for i in range(steps_test):
            test_batch_x, test_batch_y = next(data_gen_test)
            # save true_value(targets) for plotting a graph in main
            true_values += list(test_batch_y)
            test_batch_y = np.array(test_batch_y).reshape(-1, 1)
            test_predict = self.session.run(self._Y_pred, feed_dict={self._X: test_batch_x})
            rmse_val = self.session.run(self._rmse, feed_dict={
                self._targets: test_batch_y, self._predictions: test_predict})

            rmse_list = np.append(rmse_list,rmse_val)
            test_predict_list = np.append(test_predict_list, test_predict)

        print("RMSE : {}".format(rmse_list.mean()))
        return test_predict_list

    def predict_once(self, test_x):
        test_predict = self.session.run(self._Y_pred, feed_dict={self._X: test_x})
        return test_predict


