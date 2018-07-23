import numpy as np
import random
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
class Unit:
    #param: label : int
    def __init__(self,vector,label):
        self.value = vector
        self.label = label
    def get_label(self):
        return self.label
class Data_set:
    def __init__(self,data,label):
        Data_sets = []
        for i in range(len(data)):
            unit = Unit(data[i],label)
            Data_sets.append(unit)
        self.data_set = Data_sets
    def draw(self):
        data = self.data_set
class PriorityQueue:
    def __init__(self,k):
        queue = {}
        self.length = k
        self.queue = queue
    def add(self,elements):
        if (elements in (self.queue.keys())==True):
            pass
        else:
            if (len(self.queue) == self.length):
                self.queue = sorted(self.queue.keys())
                self.queue.pop()
#calculate distance between two points
def Eliucdean(X1,X2):
    return np.linalg.norm(X1-X2)

def KNN(k,data,new_data):
    data_set = data.data_set
    dists = np.zeros((len(data_set),1))
    for i in range(len(data_set)):
        dists[i] = Eliucdean(data_set[i],new_data)
if __name__ == "__main__":
    mu = np.array([[1, 5]])
    Sigma = np.array([[1, 0.5], [1.5, 3]])
    R = cholesky(Sigma)
    s = np.dot(np.random.randn(100, 2), R) + mu
    mu = np.array([[5, 10]])
    Sigma = np.array([[1, 0.5], [1.5, 3]])
    R = cholesky(Sigma)
    s1 = np.dot(np.random.randn(100, 2), R) + mu
    data_set1 = Data_set(s,0)
    data_set2 = Data_set(s1,1)
    test = np.vstack((s, s1))
    plt.plot(data_set1.data_set[:,0],data_set1.data_set[:,1],'+')
    plt.show()