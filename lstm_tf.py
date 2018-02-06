# coding=utf-8
import numpy as np
import tensorflow as tf
import datetime
import os

class LSTM:

    def __init__(self, session: tf.Session, x_window_size: int, ncols: int,
               dirname_save_model: str, name: str="main") -> None:
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
        self.save_path = dirname_save_model
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

    def train(self, epochs, steps_per_epoch, data_gen_train, save=False):
        self.session.run(tf.global_variables_initializer())
        step_loss=0

        for i in range(epochs):
            for j in range(steps_per_epoch):
                batch_x, batch_y = next(data_gen_train)

                # shuffle the batch
                p = np.random.permutation(len(batch_x))
                batch_x, batch_y = batch_x[p], batch_y[p]

                batch_y = np.array(batch_y).reshape(-1, 1)
                _, step_loss = self.session.run([self._train, self._loss], feed_dict={
                    self._X: batch_x, self._Y: batch_y})
                print("[epoch: {}, step: {}] loss: {}".format(i, j, step_loss))

            if save:
                # save model at every epoch
                self.save_model('epoch{}_loss{:.2e}'.format(i, step_loss), write_meta_graph=(i==0))

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

    def predict_sequences_multiple(self, data, window_size, prediction_len, latest=True):
        # Before shifting prediction run forward by prediction_len steps,
        # predict sequence of prediction_len steps
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.predict_once(curr_frame[np.newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)

        if latest:
            curr_frame = data[-1]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.predict_once(curr_frame[np.newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
            prediction_seqs.append(predicted)

        return prediction_seqs

    def save_model(self, modelname='', write_meta_graph=False):
        saver = tf.train.Saver()
        if modelname == '':
            # make today's file name
            today = datetime.date.today()
            today = today.strftime('%y%m%d')
            modelname = 'model' + today

        save_path = 'saved_models/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path_full = os.path.join(save_path, modelname)
        saver.save(self.session, save_path_full, write_meta_graph=write_meta_graph)

    def load_model(self, modelname):
        load_path = 'saved_models/'
        meta_filename = modelname + '.meta'
        ckpt_filename = modelname

        # meta path formatting : saved_model/modelname.meta
        meta_path_full = os.path.join(load_path, meta_filename)
        # ckpt path formatting : saved_model/modelname (error occur if + '.ckpt')
        ckpt_path_full = os.path.join(load_path, ckpt_filename)

        print('>Load checkpoint : {}'.format(ckpt_path_full))
        saver = tf.train.Saver()
        # tf.reset_default_graph()
        # saver = tf.train.import_meta_graph(meta_path_full)
        print(tf.train.latest_checkpoint(load_path))
        saver.restore(self.session, ckpt_path_full)

