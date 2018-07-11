from sklearn.datasets import load_boston
import numpy as np
import math
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import pandas as pd
import random
def normalization(xis):
    for i in range(xis.shape[1]):
        # print(xis[:,i])
        mean = np.mean(xis[:,i])
        #judge if the variance of this column is 0 or not 
        std_variance = np.std(xis[:,i],ddof=1)
        if(std_variance==0):
            pass
        else:
            xis[:,i] = (xis[:,i]-mean)/np.abs(std_variance)
    return xis
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
def mean_preprocessing(data_set):
    imp = Imputer(missing_values='NaN',strategy='mean',axis=1)
    return imp.transform(data_set)
def PCA(eigenvalue,eigenvector,xis):
    total_value = 0
    for i in range(len(eigenvalue)):
        total_value+=eigenvalue[i]
    ratio = 0
    total_real_value = 0
    new_xis = []
    rows = 0
    while (ratio != 0.8):
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
def data_pruning_for_school_explorer():
    data_set = pd.read_csv("2016 School Explorer.csv")
    data_set.drop(data_set.columns[0:6],axis = 1,inplace = True)
    data_set.drop("Address (Full)",axis = 1,inplace = True)
    data_set.drop("City",axis = 1,inplace = True)
    data_set.drop("Collaborative Teachers Rating",axis = 1,inplace = True)
    data_set.drop("Supportive Environment Rating",axis = 1,inplace = True)
    data_set.drop("Effective School Leadership Rating",axis = 1,inplace = True)
    data_set.drop("Strong Family-Community Ties Rating",axis = 1,inplace = True)
    data_set.drop("Trust Rating",axis = 1,inplace = True)
    data_set.drop("Student Achievement Rating",axis = 1,inplace = True)
    data_set.drop("Grades",axis = 1,inplace = True)
    data_set.drop("Community School?",axis = 1,inplace = True)
    data_set.drop("Student Attendance Rate",axis = 1,inplace = True)
    data_set.drop("Percent of Students Chronically Absent",axis = 1,inplace = True)
    data_set.drop("Rigorous Instruction Rating",axis = 1,inplace = True)
    data_set.drop("Grade Low",axis = 1,inplace = True)
    data_set.drop("Grade High",axis = 1,inplace=True)
    data_set = list(data_set.values)
    for i in range(len(data_set)):
        if_nan = (data_set[i][5]==data_set[i][5])
        if (if_nan==False):
            data_set[i][5] = 0.0
        else:
            data_set[i][5]= data_set[i][5].replace("$","")
            data_set[i][5]= float(data_set[i][5].replace(",",""))
        #from column 7 to column 18
        for j in range(6,18):
            #数据集的索引7-18需要去掉百分符号并且转化为float
            if_nan1 = (data_set[i][j]==data_set[i][j])
            if(if_nan1==False):
                data_set[i][j] = 0.0
            else:
                data_set[i][j]=float(data_set[i][j].replace("%",""))/100 
    data_set = np.array(data_set)
    data_set = normalization(data_set)
    data_set = mean_preprocessing(data_set)
    return data_set
def k_means_clustering(k,data_set):
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
        if (step>3):
            break
        else:
            step+=1
    def cal_cost(cluster,initial_center):
        cost = 0.0
        length = 0
        for i in range(len(initial_center)):
            for j in range(len(clusters[i])):
                cost+=math.pow(np.linalg.norm(clusters[i][j]-initial_center[i]),2)
                length+=1
        return cost/length
    plt.plot(data_set[:, 0], data_set[:, 1], '+')
    for i in range(len(initial_center)):
        plt.plot(initial_center[i][0],initial_center[i][1],'x')
    plt.show()
    return cal_cost(clusters,initial_center)
if __name__=="__main__":
    
    data_set = data_pruning_for_school_explorer()
    C = get_C(data_set)
    eigenvalue,eigenvector = get_eigen(C)
    # eigenvalue = np.array(eigenvalue,dtype=float)
    #do principle component analysis
    new_data_set,eigenvector1 = PCA(eigenvalue,eigenvector,data_set)
    new_dimension_data = get_new_points(new_data_set,eigenvector1)
    cost = []
    for i in range(1,6):
        cost1 = k_means_clustering(i,new_dimension_data)
        cost.append(cost1)
    print(cost)
    plt.plot(cost)
    # for i in range(len(cost)):
    #     plt.plot(i,cost[i],'.')
    plt.show()
    # plt.plot(data1[:,0],data1[:,1],'+')
    # plt.show()


