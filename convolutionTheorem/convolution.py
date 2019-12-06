# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True
})

# %%


def movingAverage(series, window):
    '''
    args -
    ----
    series(Array/pandas Series) = Noisy data,
    window(int) = smoothing window

    method -
    ------
    Discrete Convolution to merge two series

    Returns -
    -------
    smoothingSingal(numpy array) = Smoothing Signal,
    (numpy array) = Smooth data
    '''
    smoothingSignal = np.ones(window) / float(window)
    return smoothingSignal, np.convolve(series, smoothingSignal, 'same')


# %%
##########################################################
################## Importing Data ########################
##########################################################
df = pd.read_excel('noisyData.xlsx')
df.head()

smoothingSignal, convResult = movingAverage(df['Series'], 10)
# %%
##########################################################
######## plotting data (Noisy and Smooth) ################
##########################################################
plt.plot(df['Series'], c='b', label='Original Noisy Data')
plt.plot(smoothingSignal, c='c', label='Smoothing Signal')
plt.plot(convResult, c='y', label='Smoothed Data')
plt.legend()
plt.show()


# %%
x1 = [1, 2, -2]
x2 = [2, 0, 1]

convResult = np.convolve(x1, x2, 'full')
# %%
print(convResult)

# %%
plt.plot(x1, c='c')
plt.plot(x2, c='m')
plt.plot(convResult, c='b')
plt.show()

# %%
