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

def reject_outliers(data, m = 2.):
    if isinstance(data, list):
        data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


training_loss = np.array(training_loss, dtype=float)
training_loss_idx = np.array(training_loss_idx, dtype=float)
validation_loss = np.array(validation_loss)
validation_loss_idx = np.array(validation_loss_idx)

# Prevent discontinuities from drawing big horizontal lines.
pos = np.where(training_loss_idx[1:] != training_loss_idx[:-1] + 1)[0]
training_loss = np.insert(training_loss, pos, np.nan)
training_loss_idx = np.insert(training_loss_idx, pos, np.nan)

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
fig, ax = plt.subplots(1, 1)

#ax1.plot(lr_steps, lr)

ax.set_title("MotionCNN Training Progress")
ax.set_ylabel("Learning Rate")

legend = list()
try:
    ma_n = 500
    training_loss_ma = moving_average(training_loss, ma_n)
    ax.plot(training_loss_idx, training_loss_ma)
    legend.append(f"Training Loss (MA{10*ma_n})")
except Exception as err:
    print(f"Failed to plot training_loss_ma: {err}")
ax.plot(validation_loss_idx, validation_loss)
legend.append("Validation Loss")
try:
    validation_loss_best = [min(validation_loss[:i+1]) for i in range(len(validation_loss))]
    # ^^ Painfully inefficient, but still fast at the scale we're dealing with.
    ax.plot(validation_loss_idx, validation_loss_best)
    legend.append("Best Validation Loss")
except Exception as err:
    print(f"Failed to plot validation_loss_best: {err}")

ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
try:
    ax.set_ylim(-1500, 1500)
except Exception as err:
    print(f"Failed to set ylim: {err}")

ax.legend(legend)

plt.show()
