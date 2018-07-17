import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky
import pandas as pd
import Data_processing
from matplotlib import animation
import random
# import sys
# sys.path.append("C:\\IntelligentDataAnalysis\\Principle_Component_Analysis")
from sklearn.preprocessing import Imputer
def sigma(init,t,v):
    return init*np.exp(-t/v)
def learningrate(init,t,tao):
    return init*np.exp(-t/tao)
def h(winner,init_nwidth,j,t,v):
    # a = np.exp(-np.power(np.linalg.norm(winner - j), 2) / sigma(init_nwidth, t, v))
    a = np.exp(-np.power(np.linalg.norm(winner - j), 2) / sigma(init_nwidth, t, v))
    print(a)
    return a
def cal_winner(xi,b):
    min_index = 0
    min_distance = np.inf
    for i in range(len(b)):
        dist = np.linalg.norm(xi - b[i])
        if (dist<min_distance):
            min_distance = dist
            min_index = i
    return min_index
def quantization_error(xis,winners):
    error = 0.0
    for i in range(len(xis)):
        error+=np.linalg.norm(xis[i]-winners[i])
    return error/len(xis)
def init_codebook_vector(M,xis):
    vectors = np.zeros((M,xis.shape[1]))
    for i in range(M):
        index = np.random.randint(1,len(xis))
        vectors[i] = xis[index]
    return vectors

def main(xis):

    initial_neighborhood_width = 1
    time_scale = 50
    learning_rate = 0.1
    codebook_vectors = init_codebook_vector(30,xis)
    plt.figure()
    plt.plot(xis[:, 0], xis[:, 1], '+')
    plt.show()
    for i in range(len(xis)):
        winner_index = cal_winner(xis[i],codebook_vectors)
        for j in range(len(codebook_vectors)):
            # codebook_vectors[winner_index] = codebook_vectors[winner_index]+learningrate(learning_rate,i,60)*(xis[i]-codebook_vectors[winner_index])
            relationship =    h(codebook_vectors[winner_index],initial_neighborhood_width,codebook_vectors[j],i,time_scale)

            codebook_vectors[j] = codebook_vectors[j]+relationship*learningrate(learning_rate,i,60)*(xis[i]-codebook_vectors[winner_index])
    plt.figure()
    plt.plot(xis[:, 0], xis[:, 1], '+')
    plt.plot(codebook_vectors[:,0],codebook_vectors[:,1],'x')
    plt.show()

def x_square():
    data = np.zeros((1000,2))
    count = 0
    for i in range(-25,25):
        value = np.power(i,2)
        for j in range(20):
            value1 = value+random.uniform(-10,23)
            data[count] = np.array([value1,i])
            count+=1
    # plt.plot(data[:,0],data[:,1],'+')
    #
    # plt.show()
    return data
if __name__=="__main__":
    # mu = np.array([[1, 5]])
    # Sigma = np.array([[1, 0.5], [1.5, 3]])
    # R = cholesky(Sigma)
    # s = np.dot(np.random.randn(1000, 2), R) + mu
    # plt.figure()
    # plt.plot(s[:,0],s[:,1],'+')
    # plt.show()
    # main(s)
    data = x_square()
    main(data)
    # data = Data_processing.data_pruning_for_school_explorer()
    # main(data)