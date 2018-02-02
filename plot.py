import matplotlib.pyplot as plt
import matplotlib.style as style
import json
import pandas as pd

style.use('seaborn')
configs = json.loads(open('configs.json').read())
data = pd.read_csv(configs['data']['filename'], index_col=0)
lasttime = data.iloc[-1].name

def plot_results(predicted_data, true_data):
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data', linewidth=1, color="black")
    plt.plot(predicted_data, label='Prediction', linewidth=1, color='orange')
    plt.legend()
    plt.show(block=False)

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(1)
    fig.suptitle('last prediction starts from '+ str(lasttime))
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data', linewidth=1.5, color="black")
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction', linewidth=1)
        plt.legend()
    plt.show()
