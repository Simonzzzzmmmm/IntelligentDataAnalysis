from sklearn.datasets import load_boston
import numpy as np
import math
import Data_processing
import matplotlib.pyplot as plt
import pandas as pd
import random
def get_C(xis):
    return (np.dot(np.transpose(xis),xis))/len(xis)
    # return np.cov(np.array(xis,dtype=float))
def get_eigen(covariance):
    return np.linalg.eig(covariance)
def show_graph(points):
    plt.figure()
    for i in range(points.shape[0]):
        plt.plot(points,'.')
def get_new_points(points,v):
    new_v = v[:,0:points.shape[1]]
    new_points = np.dot(points, np.transpose(new_v))
    return new_points
def PCA(eigenvalue,eigenvector,xis):
    total_value = 0
    for i in range(len(eigenvalue)):
        total_value+=eigenvalue[i]
    ratio = 0
    total_real_value = 0
    new_xis = []
    rows = 0
    while (ratio != 0.9):
        max_index = np.argmax(eigenvalue)
        new_xis.append(xis[:,max_index])
        #delete the dimension that has the max value, for the following argmax.
        xis = np.delete(xis,max_index,axis=1)
        eigenvalue = np.delete(eigenvalue,eigenvalue[max_index],axis=0)
        total_real_value+=eigenvalue[max_index]
        rows+=1
        if (total_real_value/total_value>=0.8):
            break
    # new_eigenvector = np.zeros((rows,rows))
    new_eigenvector = eigenvector[:rows,:rows]
    new_axis = np.transpose(np.array(new_xis))
    return new_axis,new_eigenvector

def k_means_clustering(low, median, high,k,data_set):
    #randomly choose k initial center points of clusters, tricky: let data_set be the central points
    def init(k):
        # initial_center = np.zeros((k,len(data_set[0])))
        initial_center = []
        for i in range(k):
            random_index = random.randrange(1,len(data_set))
            initial_center.append(data_set[random_index])
        return initial_center
    def Elucidean_distance(X1,X2):
        dist = np.linalg.norm(X1-X2)
        return dist
    def averagenum(num,i,k):
        nsum = 0
        for j in range(len(num[i])):
            nsum += num[i][j][k]
        nsum = nsum/len(num[i])
        return nsum
    def find_clusters(point,initial_center):
        min_index = math.inf
        min_distance = math.inf
        for i in range(len(initial_center)):
            distance = Elucidean_distance(point,initial_center[i])
            if (distance<min_distance):
                min_distance = distance
                min_index = i
        return [min_index,min_distance]
    clusters = [[] for i in range(k)]
    initial_center = init(k)
    step = 0
    while True: 
        for i in range(len(data_set)):
            #find the cluster which is nearest to the point
            min_index,min_distance = find_clusters(data_set[i],initial_center)
            #add the points to the corresponding cluster
            clusters[min_index].append(data_set[i])
        for i in range(len(initial_center)):
            #for every dimension
            for j in range(len(initial_center[0])):
                #the ith cluster's central point, change initial central points
                initial_center[i][j] = averagenum(clusters,i,j)
        if (step>5):
            break
        else:
            step+=1
    def cal_cost(cluster,initial_center):
        cost = 0.0
        size = 0.0
        for i in range(len(initial_center)):
            for j in range(len(clusters[i])):
                cost+=math.pow(np.linalg.norm(clusters[i][j]-initial_center[i]),2)
                size+=1
        return cost/size
    draw_color_graph(low,median,high,data_set,2,3)
    for i in range(len(initial_center)):
        plt.scatter(initial_center[i][2],initial_center[i][3],s = 100,c = 'k',marker='o')
    plt.scatter(initial_center[0][2], initial_center[0][3], c='k', label="Cluster point")
    plt.legend(loc='upper right')
    return initial_center,cal_cost(clusters,initial_center)
#graph with ground truth
def draw_color_graph(low,median,high,data_set,one,two):
    plt.plot(data_set[0][one], data_set[0][two], 'r.', label='low')
    for i in low:
        plt.plot(data_set[i][one], data_set[i][two], 'r.')
    plt.legend(loc='upper right')
    plt.plot(data_set[0][one], data_set[0][two], 'b.', label='median')
    for i in median:
        plt.plot(data_set[i][one], data_set[i][two], 'b.')
    plt.legend(loc='upper right')
    plt.plot(data_set[0][one], data_set[0][two], 'y.', label='high')
    for i in high:
        plt.plot(data_set[i][one], data_set[i][two], 'y.')
    plt.legend(loc='upper right')
#gtaph with clustering result
def label_clustering_graph(initial_center,data_set):
    clusters = [[] for i in range(len(initial_center))]
    def cal_cluster(data,initial_center):
        min_dist = np.inf
        min_index = np.inf
        for i in range(len(initial_center)):
            dist = np.linalg.norm(data-initial_center[i])
            if (dist<min_dist):
                min_dist = dist
                min_index = i
        return min_index
    for i in range(len(data_set)):
        index = cal_cluster(data_set[i],initial_center)
        clusters[index].append(data_set[i])
    colors = ['r','b','y','g']
    count=0
    plt.figure()
    for cluster in clusters:
        cluster = np.array(cluster)
        print(cluster)
        plt.scatter(cluster[:,2],cluster[:,3],c = colors[count],marker='.',label = 'data points')
        count += 1

    for i in range(len(initial_center)):
        plt.scatter(initial_center[i][2],initial_center[i][3],s = 100,c = 'k',marker='o')
    plt.scatter(initial_center[0][2], initial_center[0][3], c = 'k',label="Cluster point")
    plt.legend(loc='upper left')
    plt.show()
def PCA_topo_with_two_leadingeigen(dataset,vector,i,j):
    """we only need two eigen-direction, which means two lead eigen vectors"""
    plt.figure(j)
    plt.plot(dataset[:,i],dataset[:,j],'.',label = 'data points')
    plt.plot(vector[0:2,i],vector[0:2,j],'o',label = 'eigen vectors')
    plt.legend(loc = "upper left")
    plt.show()
if __name__=="__main__":
    low, median, high,data_set = Data_processing.data_pruning_for_school_explorer()
    C = get_C(data_set)
    eigenvalue,eigenvector = get_eigen(C)
    # eigenvalue = np.array(eigenvalue,dtype=float)
    #do principle component analysis
    new_data_set,eigenvector1 = PCA(eigenvalue,eigenvector,data_set)
    new_dimension_data = get_new_points(new_data_set,eigenvector1)
    # for i in range(0,3):
    #     PCA_topo_with_two_leadingeigen(new_dimension_data,eigenvector,i,i+1)
    
    # plt.figure()
    # plt.plot(eigenvalue)
    cost = []
    location = 1
    # initial_center, cost1 = k_means_clustering(low, median, high, 3, new_dimension_data)
    # label_clustering_graph(initial_center,new_dimension_data)
    # '''PCA graph based on k-means'''
    # initial_center,cost1 = k_means_clustering(low, median, high,3,new_dimension_data)
    # label_clustering_graph(initial_center,new_dimension_data)
    # '''PCA graph based on k-means'''
    # print(cost)
    # plt.figure()
    # plt.plot(cost)
    # for i in range(len(cost)):
    #     plt.plot(i,cost[i],'.')
    # plt.show()
    # plt.plot(data1[:,0],data1[:,1],'+')
    # plt.show()


