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
    plt.figure()
    draw_color_graph(low,median,high,data_set)
    for i in range(len(initial_center)):
        plt.scatter(initial_center[i][0],initial_center[i][1],s = 100,c = 'k',marker='o')
    plt.show()
    return initial_center,cal_cost(clusters,initial_center)
#graph with ground truth
def draw_color_graph(low,median,high,data_set):
    plt.plot(data_set[0][0], data_set[0][1], 'r.', label='low')
    for i in low:
        plt.plot(data_set[i][0],data_set[i][1],'r.')
    plt.legend(loc='upper right')
    plt.plot(data_set[0][0], data_set[0][1], 'b.', label='median')
    for i in median:
        plt.plot(data_set[i][0], data_set[i][1], 'b.')
    plt.legend(loc='upper right')
    plt.plot(data_set[0][0], data_set[0][1], 'y.', label='high')
    for i in high:
        plt.plot(data_set[i][0], data_set[i][1], 'y.')
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
        plt.scatter(cluster[:,0],cluster[:,1],c = colors[count],marker='.')
        count += 1
    for i in range(len(initial_center)):
        plt.scatter(initial_center[i][0],initial_center[i][1],s = 100,c = 'k',marker='o')
    plt.show()

if __name__=="__main__":
    low, median, high,data_set = Data_processing.data_pruning_for_school_explorer()
    C = get_C(data_set)
    eigenvalue,eigenvector = get_eigen(C)
    # eigenvalue = np.array(eigenvalue,dtype=float)
    #do principle component analysis
    new_data_set,eigenvector1 = PCA(eigenvalue,eigenvector,data_set)
    new_dimension_data = get_new_points(new_data_set,eigenvector1)


    # plt.figure()
    # plt.plot(eigenvalue)
    # cost = []
    # location = 1
    # initial_center, cost1 = k_means_clustering(low, median, high, 3, new_dimension_data)
    # label_clustering_graph(initial_center,new_dimension_data)
    # for i in range(1,5):
    #     plt.subplot(2,2,location)
    #     plt.title("k="+str(location))
    #     initial_center,cost1 = k_means_clustering(low, median, high,i,new_dimension_data)
    #     cost.append(cost1)
    #     location+=1
    # plt.show()
    # print(cost)
    # plt.figure()
    # plt.plot(cost)
    # for i in range(len(cost)):
    #     plt.plot(i,cost[i],'.')
    # plt.show()
    # plt.plot(data1[:,0],data1[:,1],'+')
    # plt.show()


