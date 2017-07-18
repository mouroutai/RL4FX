import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mysql.connector
import pandas_datareader as web
import datetime
#------------------------------------------------------------------------------
# DEFINITION
#------------------------------------------------------------------------------
class RLMomentum():
    def __init__(self, data, testData):
        #self.data = pd.read_csv(datapath, header=None)
        self.data = pd.DataFrame(data)
        self.testData = pd.DataFrame(testData)

        self.ret = self.data / self.data.shift(1) - 1
        self.ret = self.ret.fillna(0)
        self.testRet = self.testData / self.testData.shift(1) - 1
        self.testRet = self.testRet.fillna(0)

        self.window_short = 20
        self.window_long = 60
        self.samples = len(self.data)
        self.testSamples = len(self.testData)
        self.states = 15
        self.actions = 3 #long, flat, short
        self.epsilon = 0.1
        self.gamma = 0.9 #discount factor
        self.mc = 10000 #Monte Carlo
        self.isCompound = False

        self.q = np.zeros((self.states, self.states, self.actions))
        self.rewards = np.zeros((self.states, self.states, self.actions))
        self.count = np.zeros((self.states, self.states, self.actions), dtype = np.int16)
        self.isVisited = np.zeros((self.states, self.states, self.actions), dtype = np.bool)

        self.momentum = np.zeros(self.samples)
        self.testMomentum = np.zeros(self.testSamples)
        self.cumulative = np.zeros(self.mc)


    def init(self):
        self.count = np.zeros((self.states, self.states, self.actions), dtype = np.int16)
        self.isVisited = np.zeros((self.states, self.states, self.actions), dtype = np.bool)

    def currentState(self, signal):
        signal = float(signal)
        sep = np.linspace(-1, 1, self.states-1)
        return sum(sep < signal)

    def selectAction(self, state_short, state_long, threshold):
        if (self.q[state_short, state_long, :]==0).sum() == self.actions:
            #if all action-values are 0
            return np.random.randint(0, self.actions)
        else:
            #Epsilon-Greedy
            if np.random.random(1) < threshold:
                return np.random.randint(0, self.actions)
            else:
                return np.argmax(self.q[state_short, state_long, :])

    def actionToPosition(self, action):
       return action -1

    def updateRewards(self, reward, state_short, state_long, action):
        self.isVisited[state_short, state_long, action] = True
        self.rewards = self.rewards + reward.values * (self.gamma ** self.count)
        self.count = self.count + self.isVisited

    def updateQ(self, itr):
        self.q = (self.q * itr + self.rewards) / (itr + 1)

    def episode(self):
        for i in range(self.samples - 1):
            if i <= self.window_long - 1:
                self.momentum[i] = self.ret.ix[i]
            else:
                sub_short = self.momentum[i - self.window_short : i - 1]
                sub_long = self.momentum[i - self.window_long : i - 1]

                #state = annualized Sharpe ratio
                state_short = self.currentState( np.mean(sub_short) / np.std(sub_short) * np.sqrt(252) )
                state_long = self.currentState( np.mean(sub_long) / np.std(sub_long) * np.sqrt(252) )

                action = self.selectAction(state_short, state_long, self.epsilon)

                if self.isCompound:
                    reward = np.log( 1 + self.ret.ix[i + 1] * self.actionToPosition(action) )
                else:
                    reward = self.ret.ix[i + 1] * self.actionToPosition(action)

                self.updateRewards(reward, state_short, state_long, action)

                self.momentum[i] = self.ret.ix[i + 1] * self.actionToPosition(action)

    def test(self):
        for i in range(self.testSamples - 1):
            if i <= self.window_long - 1:
                self.testMomentum[i] = self.testRet.ix[i]
            else:
                sub_short = self.testMomentum[i - self.window_short : i - 1]
                sub_long = self.testMomentum[i - self.window_long : i - 1]

                #state = annualized Sharpe ratio
                state_short = self.currentState( np.mean(sub_short) / np.std(sub_short) * np.sqrt(252) )
                state_long = self.currentState( np.mean(sub_long) / np.std(sub_long) * np.sqrt(252) )

                action = self.selectAction(state_short, state_long, 0)

                self.testMomentum[i] = self.testRet.ix[i + 1] * self.actionToPosition(action)


    def monteCarlo(self):
        for i in range(self.mc):
            self.init()
            self.episode()
            self.cumulative[i] = np.mean(self.momentum) * 252
            self.updateQ(i)




        #test strategy
        self.test()

        testRet = np.mean(self.testMomentum) * 252
        testVol = np.std(self.testMomentum) * np.sqrt(252)
        testSR = testRet / testVol

        fig, ax = plt.subplots()
        ax.set_xlabel("Episode step")
        ax.set_ylabel("Strategy, " + "Ret:" + str(np.round(testRet, 2)) + " Vol:" + str(np.round(testVol, 2)) + " SR:" + str(np.round(testSR, 2)) )
        ax.set_title("Strategy comparison")
        plt.plot(100*(1 + self.testRet).cumprod(), label="testRet")
        plt.plot(100*(1 + self.testMomentum).cumprod(), label="testMomentum")
        plt.legend(loc="best")
        plt.grid(True)




        #plot Q-value matrix
        x = np.linspace(0, self.states-1, self.states)
        y = np.linspace(0, self.states-1, self.states)
        x,y = np.meshgrid(x, y)

        fig, ax = plt.subplots()
        ax.set_xlabel("state_short")
        ax.set_ylabel("state_long")
        ax.set_title("optimal position")
        #axis = number of states
        cax = ax.imshow(np.argmax(self.q, axis = 2), interpolation='nearest', cmap=cm.coolwarm)
        cbar = fig.colorbar(cax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(['short', 'flat', 'long'])# vertically oriented colorbar


        #plot cumulative return
        fig, ax = plt.subplots()
        ax.set_xlabel("number of Monte Carlo")
        ax.set_ylabel("Annualized cumulative return[%]")
        ax.set_title("Annualized cumulative return")
        plt.plot(pd.expanding_mean(self.cumulative))
        plt.grid(True)

        plt.show()


#------------------------------------------------------------------------------
# MAIN
#------------------------------------------------------------------------------
start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2015, 12, 31)
start2 = datetime.datetime(2016, 1, 1)
end2 = datetime.datetime(2016, 12, 31)
data = web.DataReader("DEXJPUS","fred",start, end)
test = web.DataReader("DEXJPUS", "fred",start2, end2)
m = RLMomentum(data, test)
m.monteCarlo()
