#!/usr/bin/env python
# coding: utf-8

# ## Import Functions

# In[4]:


import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
import numpy as np
import pandas as pd
import math
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from mpl_finance import candlestick2_ohlc
import datetime
from collections import deque


# ## Defining The Agent

# In[2]:


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 3 # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._model()
        #self.model = load_model("/Users/maharshichattopadhyay/Documents/Study/Major_Project/Model/" + model_name) if is_eval else self._model()
    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model
    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])
    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    #def load_checkpoint(self, path):
        #self.model.load_weights(path)


# ## Plot Function

# In[11]:


def plot(csv,sell_time):
    df=pd.read_csv(csv)
    date=df['Date']
    open=df['Open']
    high=df['High']
    low=df['Low']
    close=df['Close']
    fig=plt.figure(figsize=(7,5))
    graph=fig.add_subplot(1,1,1)
    graph.plot(date,close,color='red')
    candlestick2_ohlc(graph,open,high,low,close,colorup="blue",colordown="red",width=2,alpha=0.5)


# ## Math-Functions
# #### Intuition: This Function Defines How The Input Will Be Processed and The Output Will Be Shown

# In[3]:


def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))
def getStockDataVec(CSV):
    vec = []
    lines = open(CSV, "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[8]))# Selecting The Close Price And Multiplying it By 1000
    return vec
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


# ## Importing From GPU To CPU

# In[4]:


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)
    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])
     # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)


# ## Training The Model

# In[5]:


stock_name="/Users/maharshichattopadhyay/Documents/Study/Major_Project/DataSet/Final_Dataset/Final_Data_MCB_Train.csv"
window_size=10
episode_count = 2
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
#path = '/Users/maharshichattopadhyay/Desktop/Model/No_Of_Episodes_380.h5'
#agent.load_checkpoint(path)
for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    for t in range(l):
        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1: # buy
            agent.inventory.append(data[t])
            print("Buy: " + formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
        if e % 10 == 0:
            filepath="/Users/maharshichattopadhyay/Documents/Study/Major_Project/Model/No_Of_Episodes_" + str(e)+".hdf5"
            agent.model.save(filepath)


# ## Testing The Model

# In[9]:


stock_name="/Users/maharshichattopadhyay/Documents/Study/Major_Project/DataSet/Final_Dataset/Final_Data_MCB_Test.csv"
model_name = "/Users/maharshichattopadhyay/Documents/Study/Major_Project/Model/No_Of_Episodes_100.hdf5"
model = agent.model.load_weights(model_name)
#model = agent.model.load_weights(model_name)[-4]
window_size = agent.model.layers[0].input.shape.as_list()[1]
agent = Agent(window_size, True, model_name)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
state = getState(data, 0, window_size + 1)
total_profit = 0
sell_time=[]
agent.inventory = []
for t in range(l):
    action = agent.act(state)
    # sit
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0
    if action == 1: # buy
        agent.inventory.append(data[t])
        print("Buy: " + formatPrice(data[t]))
    elif action == 2 and len(agent.inventory) > 0: # sell
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
        sell_time=formatPrice(data[t])
    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state
    if done:
        print("--------------------------------")
        print(stock_name + " Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")


# In[5]:


#plot(stock_name,sell_time)


# In[ ]:




