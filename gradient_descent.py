import numpy as np
import pandas as pd

# data_set = pd.read_csv('travel insurance.csv',usecols=['Age' ,'Gender' ,'Claim' ])
# data_set['Gender'].replace({'F':0 , 'M':1},inplace=True)
# data_set['Claim'].replace({'No':0 , 'Yes':1},inplace=True)

def log_loss(y_true, y_predict):
    eplison = 1e-15
    y_predicted_new = [max(i, eplison) for i in y_predict]
    y_predicted_new = [max(i, 1 - eplison) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradient_decent(x, y, y_true, epochs):  # x, y ,y_true in csv file
    w1 = w2 = 1
    count = 0
    bais = 0
    learning_rate = 0.4
    n = len(x)
    for i in range(epochs):
        count += 1
        sum_of_all = w1 * x + w2 * y + bais
        y_predected = sigmoid(sum_of_all)
        loss = log_loss(y_true, y_predected)

        # deratives
        w1_d = 1 / n * np.dot(np.transpose(x), (y_true - y_predected))
        w2_d = 1 / n * np.dot(np.transpose(y), (y_true - y_predected))
        bais_d = np.mean(y_true - y_predected)

        w1 = w1 - learning_rate * w1_d
        w2 = w2 - learning_rate * w2_d
        bais = bais - learning_rate * bais_d

        print(f'loop {count} ===>  w1 is --> {w1} === w2 is --> {w2} === bais is --> {bais}')

    return w1, w2, bais



# x = data_set['Gender']
# y = data_set['Age']
# real = data_set['Claim']
