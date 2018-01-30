# coding=utf-8
import tensorflow as tf
import numpy as np
import etl_tf, lstm_tf
import h5py
import json
import time
import matplotlib.pyplot as plt

tf.set_random_seed(777) # for reproducibility

def plot_results(predicted_data, true_data):
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Before shifting prediction run forward by prediction_len steps,
    # predict sequence of prediction_len steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict_once(curr_frame[np.newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


configs = json.loads(open('configs.json').read())
tstart = time.time()

dl = etl_tf.ETL(
    filename_in = configs['data']['filename'],
    filename_out = configs['data']['filename_clean'],
    batch_size = configs['data']['batch_size'],
    x_window_size = configs['data']['x_window_size'],
    y_window_size = configs['data']['y_window_size'],
    y_col = configs['data']['y_predict_column'],
    filter_cols = configs['data']['filter_columns'],
    normalize = True
)

# 새로운 데이터(window_size 사용 시 주석 해제하여야 함
# dl.create_clean_datafile()

print('> Generating clean data from:', configs['data']['filename_clean'],
      'with batch_size:', configs['data']['batch_size'])

with h5py.File(configs['data']['filename_clean'], 'r') as hf:
    nrows = hf['x'].shape[0]
    ncols = hf['x'].shape[2]

ntrain = int(configs['data']['train_test_split'] * nrows)
steps_per_epoch = int(ntrain / configs['data']['batch_size'])

# ntrain를 batch_size의 배수로 만들어 경계 값을 명확히 한다.
ntrain = steps_per_epoch * configs['data']['batch_size']
print('> Clean data has', nrows, 'data rows. Training on', ntrain, 'rows with', steps_per_epoch, 'steps-per-epoch')

# Building a model
sess =tf.Session()
model = lstm_tf.LSTM(sess, configs['data']['x_window_size'], ncols)
sess.run(tf.global_variables_initializer())

# Train the model
data_gen_train = dl.generate_clean_data(0, ntrain)
model.train(configs['model']['epochs'], steps_per_epoch, data_gen_train)

ntest = nrows - ntrain
steps_test = int(ntest / configs['data']['batch_size'])
print('> Testing model on', ntest, 'data rows with', steps_test, 'steps')

# Test 1. Predict t+1 close price
data_gen_test = dl.generate_clean_data(ntrain, -1)
true_values = []
predictions = model.predict(steps_test, data_gen_test, true_values)

# Save our predictions
with h5py.File(configs['model']['filename_predictions'], 'w') as hf:
    dset_p = hf.create_dataset('predictions', data=predictions)
    dset_y = hf.create_dataset('true_values', data=true_values)

# plot a subset of the data
plot_results(predictions[:800], true_values[:800])

# Test 2. Predict t+1, t+2, ... , t+50 close prices
# We are going to cheat a bit here
# and just take batch_size data from the testing generator
# and predict that data in its whole
data_gen_test = dl.generate_clean_data(ntrain, -1)
data_x, true_values = next(data_gen_test)
prediction_len = 50  # number of steps to predict into the future

predictions_multiple = predict_sequences_multiple(
    model,
    data_x,
    data_x.shape[1],
    prediction_len
)

plot_results_multiple(predictions_multiple, true_values, prediction_len)