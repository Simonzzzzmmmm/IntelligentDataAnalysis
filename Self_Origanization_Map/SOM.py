import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import cholesky
import pandas as pd
import PCA
import Data_processing

import random
def data():

    data = []
    height = 0
    for i in range(20):
        for j in range(20):
            data.append([i,j,height])
        height+=1
    return np.array(data)
from sklearn.preprocessing import Imputer
def sigmaf(init,t,v):
    return init*np.exp(-t/v)
def learningrate(init,t,tao):
    return init*np.exp(-t/tao)
def h(winner,init_nwidth,j,sigma,t,v):
    # a = np.exp(-np.power(np.linalg.norm(winner - j), 2) / sigma(init_nwidth, t, v))
    a = np.exp(-np.power(np.linalg.norm(winner - j), 2) / np.power(sigmaf(init_nwidth,t,v),2))
    return a
def cal_winner(xi,b):
    min_index = np.inf
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
def square_codebook_vector(x,y):
    vectors = []
    for i in range(x):
        for j in range(y):
            points = [i,j,0]
            vectors.append(points)
    return np.array(vectors,dtype=float)
def square_main(data,vector):
    initial_neighborhood_width = 5
    time_scale = 1
    tao = 50
    learning_rate = 0.05
    sigma = 2
    iteration = 5
    # ax.scatter(data[:, 0], data[:, 1],data[:,2], '.')

    # ax.scatter(data[:, 0], data[:, 1],data[:,2], '.',label = "data")
    for k in range(iteration):
        print("k="+str(k))
        for i in range(len(data)):
            winner_index = cal_winner(data[i], vector)
            for j in range(len(vector)):
                relationship = h(vector[winner_index], initial_neighborhood_width, vector[j],sigma,k,time_scale)
                gradient = (data[i] - vector[j])
                learn = learningrate(learning_rate, k, tao)
                vector[j] = vector[j] + relationship * learn * gradient
    # ax.scatter(vector[:, 0], vector[:, 1], vector[:, 2], '.')
    # ax.plot(vector[:,0],vector[:,1],vector[:,2],c = "y",label = "code book vectors")
    # plt.legend(loc='upper right')
    # plt.show()
    return vector,data
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
'''SOM projection to the 2D dimension space'''
def get_random_points(x,y,r):
    random_x = random.uniform(x-r,x+r)
    p_n = random.uniform(0,1)
    if p_n<0.5:
        random_y = x-np.power(np.power(r,2)-np.power((random_x-x),2),0.5)
    else:
        random_y = x + np.power(np.power(r, 2) - np.power((random_x - x), 2), 0.5)
    return [random_x,random_y]
#project data into 2D space based on code book vectors
def SOM_topo(data,vectors):
    distances = [[] for i in range(len(vectors))]
    points = [[] for i in range(len(vectors))]
    for i in range(len(data)):
        index = cal_winner(data[i],vectors)
        dist = np.linalg.norm(vectors[index]-data[i])
        distances[index].append(dist)
    twoD_vectors = get_codebook_coordinates(vectors)
    '''for every vectors find the distance between its and the points.'''
    for i in range(len(twoD_vectors)):
        print(twoD_vectors)
        for distance in distances[i]:
            points[i].append(get_random_points(twoD_vectors[i][0],twoD_vectors[i][1],distance))
    output = np.zeros((len(points),2))
    plt.figure()
    for point in points:
        point = np.array(point)
        if (len(point)!=0):
            point = np.array(point)
            plt.plot(point[:,0],point[:,1],'g.')
    plt.plot(points[0][0], points[0][1], 'g.',label = 'data points')
    plt.plot(twoD_vectors[:,0],twoD_vectors[:,1],'y.',label = 'code book vectors')
    plt.legend(loc="upper left")
    plt.show()

    return points,twoD_vectors
#project codebook vectors into 2D space
def get_codebook_coordinates(vectors):
    twoD_vectors = []
    twoD_vectors.append([vectors[0][0],vectors[0][1]])
    for i in range(1,len(vectors)):
        dist = np.linalg.norm(vectors[i]-vectors[i-1])
        new_vector_points = get_random_points(vectors[i-1][0],vector[i-1][1],2*dist)
        twoD_vectors.append([new_vector_points[0],new_vector_points[1]])
    return np.array(twoD_vectors)

if __name__=="__main__":
    # mu = np.array([[1, 5]])
    # Sigma = np.array([[1, 0.5], [1.5, 3]])
    # R = cholesky(Sigma)
    # s = np.dot(np.random.randn(1000, 2), R) + mu
    # plt.figure()
    # plt.plot(s[:,0],s[:,1],'+')
    # plt.show()
    # main(s)
    fig = plt.figure()
    ax = Axes3D(fig)
    # data = data()
    # ax.scatter(data[:,0],data[:,1],data[:,2])
    # data = Data_processing.data_pruning_for_school_explorer()
    # vectors = init_codebook_vector(20,data)
    # square_main(data,vectors)

    low, median, high,data = Data_processing.data_pruning_for_school_explorer()
    vector,data = square_main(data,init_codebook_vector(4,data))
    SOM_topo(data,vector)
    print("PCA+SOM")
    '''Q 5.4 first PCA then SOM vs only SOM'''
    C = PCA.get_C(data)
    eigenvalue, eigenvector = PCA.get_eigen(C)
    # eigenvalue = np.array(eigenvalue,dtype=float)
    # do principle component analysis
    new_data_set, eigenvector1 = PCA.PCA(eigenvalue, eigenvector, data)
    new_dimension_data = PCA.get_new_points(new_data_set, eigenvector1)
    vector1,data1 = square_main(new_dimension_data, init_codebook_vector(4, new_dimension_data))
    print(vector1)
    SOM_topo(data1,vector1)
    '''Q 5.4 first PCA then SOM vs only SOM'''
    # '''SOM topological graph'''
    # points,twoD_vector = SOM_topo(data,vector)
    # output = []
    # for i in range(len(points)):
    #     for j in range(len(points[i])):
    #         output.append(points[i][j])
    # output = np.array(output)
    # # initial_center, cost1 = PCA.k_means_clustering(low, median, high, 3, output)    # print(vectors)
    # # PCA.label_clustering_graph(initial_center,output)
    # plt.figure()
    # plt.scatter(output[:,0],output[:,1],marker = '.',label = "data points")
    # plt.scatter(twoD_vector[:,0],twoD_vector[:,1],marker='o',s = 20,label = 'code book vectors')
    # plt.legend(loc = "upper left")
    # plt.show()



    # data_set = Data_processing.data_pruning_for_school_explorer()
    # C = PCA.get_C(data_set)
    # eigenvalue, eigenvector = PCA.get_eigen(C)
    # # eigenvalue = np.array(eigenvalue,dtype=float)
    # # do principle component analysis
    # new_data_set, eigenvector1 = PCA.PCA(eigenvalue, eigenvector, data_set)
    # new_dimension_data = PCA.get_new_points(new_data_set, eigenvector1)
    # ax.scatter(new_dimension_data[:,0],new_dimension_data[:,1],new_dimension_data[:,2],'.')
    # ax.scatter(eigenvector[:,0],eigenvector[:,1],eigenvector[:,2],'.')
    # plt.show()