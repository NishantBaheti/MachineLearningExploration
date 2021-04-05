# %%
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize' : (12,6),
    'axes.grid' : True
})
# %%

df = pd.read_csv('selectedData.csv')
df.dropna(inplace=True)

df = df[[ 'DepTime', \
       'CRSDepTime', 'ArrTime', 'CRSArrTime', \
       'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', \
       'DepDelay', 'Distance','CarrierDelay', \
       'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay' \
   ]]

#%%
plt.plot(df['LateAircraftDelay'],c='g')
plt.plot(df['CarrierDelay'],c='b')
# plt.plot(df['DepDelay'],c='r')
plt.title('Comparison')
plt.legend()
plt.show()


# %%
######################################################################
###################### Feature and Target DataFrames #################
######################################################################
featureDf = df[['DepTime', \
       'CRSDepTime', 'ArrTime', 'CRSArrTime', \
       'ActualElapsedTime', 'CRSElapsedTime', 'AirTime', 'ArrDelay', \
       'DepDelay', 'Distance','CarrierDelay', \
       'WeatherDelay', 'NASDelay', 'SecurityDelay'\
	]]
targetDf = df['LateAircraftDelay']
#%%

#%%
###################################################################
################## Creating Training and Testing Dataset ##########
###################################################################
trainFeatureDf = featureDf[:50000]
trainTargetDf = targetDf[:50000]
testFeatureDf = featureDf[50000:]
testTargetDf = targetDf[50000:]

#%%
# %%
############################################################
############## Generalized Linear Model ####################
############################################################
gamma_model = sm.GLM(trainTargetDf,trainFeatureDf, family=sm.families.Gamma())
trainedModel = gamma_model.fit()


# %%
plt.plot(testTargetDf,c='y')
plt.plot(trainedModel.predict(testFeatureDf),c='b')
plt.show()

# %%
