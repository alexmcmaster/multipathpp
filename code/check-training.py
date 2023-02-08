#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
import numpy as np

fn = "train.log"

training_loss = list()
training_loss_idx = list()
validation_loss = list()
validation_loss_idx = list()
with open("train.log") as f:
    c = csv.reader(f)
    for row in c:
#        if "Loaded config" in row[0]:
#            training_loss = list()
#            training_loss_idx = list()
#            validation_loss = list()
#            validation_loss_idx = list()
#            continue
        if "Step" not in row[0]:
            continue
        step = int(row[0].split(" ")[-1])
        if step == 0 and len(training_loss) == 0:
            training_loss = list()
            training_loss_idx = list()
            validation_loss = list()
            validation_loss_idx = list()
            continue
        if len(row) < 2:
            continue
        if "train" in row[1]:
            training_loss.append(min(1e6, float(row[1].split(" ")[-1])))
            training_loss_idx.append(step)
        if "val" in row[1]:
            validation_loss.append(min(1e6, float(row[1].split(" ")[-1])))
            validation_loss_idx.append(step)

def moving_average(a, n=100) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    #return ret[n - 1:] / n
    #return np.hstack((a[:n], ret[n:] / n))
    return ret / np.hstack((np.arange(n)+1, np.full(len(ret)-n, n)))

plt.plot(training_loss_idx, moving_average(training_loss, n=1000))
#plt.plot(training_loss_idx, training_loss)
if len(validation_loss) > 50:
    plt.plot(validation_loss_idx, moving_average(validation_loss, n=5))
elif len(validation_loss) > 1:
    plt.plot(validation_loss_idx, validation_loss)
#plt.plot(training_loss_idx)

plt.ylim(1.1 * min(training_loss), 0)
#plt.ylim(0, 100000)
plt.grid()

plt.show()
