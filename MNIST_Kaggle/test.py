import tensorflow as tf
import pandas as pd
import numpy as np
import keras
df_train=pd.read_csv('train.csv')
df_features=df_train.iloc[:,1:785]
df_labels=df_train.iloc[:,0]
print(df_features.shape)
X_train=df_features[:33600,:]
X_val=df_features[33600:,:]
y_train=df_labels[:33600,:]
y_val=df_labels[33600:,:]




