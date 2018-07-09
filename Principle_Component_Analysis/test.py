from sklearn.datasets import load_boston
import numpy as np
import scipy.linalg
from sklearn import preprocessing
from numpy.linalg import cholesky
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
    new_points = np.dot(points, np.transpose(v))
    return new_points
def mean_preprocessing(data_set):
    imp = Imputer(missing_values='NaN',strategy='mean',axis=1)
    return imp.transform(data_set)
def PCA(eigenvalue,xis):
    total_value = 0
    for i in range(len(eigenvalue)):
        total_value+=eigenvalue[i]
    ratio = 0
    total_real_value = 0
    new_xis = []
    while (ratio != 0.8):
        max_index = np.argmax(eigenvalue)
        new_xis.append(xis[:,max_index])
        #delete the dimension that has the max value, for the following argmax.
        xis = np.delete(xis,max_index,axis=1)
        eigenvalue = np.delete(eigenvalue,eigenvalue[max_index],axis=0)
        total_real_value+=eigenvalue[max_index]
        if (total_real_value/total_value>=0.8):
            break
    new_axis = np.transpose(np.array(new_xis))
    return new_axis
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
        # if (isinstance(data_set[i][4], (int, float))):
        #     if (data_set[i][4]==data_set[i][4]):
        #         #数据集的索引4需要转化为float
        #         data_set[i][4] = float(data_set[i][4])
        #     else:
        #         data_set[i][4] = 1.0
        # else:
        #     data_set[i][4] = 1.0
        #数据集的索引6需要去掉美元符号和逗号并且转化为float
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
    k_points = [[]for i in range(k)]
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
    # def cal_cost(initial_center,data_set):
    #     sum = 0 
    #     for i in len(data_set):
    #         temp = data_set[i]-initial_center[i]
    #         sum += float(np.dot(np.transpose(temp),(temp)))
    #     return sum
    # 计算平均数
    def averagenum(num,j):
        nsum = 0
        for i in range(len(num)):
            nsum += num[i][j]
        return nsum / len(num)
    def find_clusters(point,initial_center):
        min_index = 9999
        min_distance = 9999
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
                initial_center[i][j] = averagenum(data1,j)
        if (step<500):
            break
        else:
            step+=1
    plt.plot(data1[:, 0], data1[:, 1], '+')
    for i in range(len(initial_center)):
        plt.plot(initial_center[i][0],initial_center[i][1],'x')
    plt.show()

if __name__=="__main__":
    
    data_set = data_pruning_for_school_explorer()
    C = get_C(data_set)
    eigenvalue,eigenvector = get_eigen(C)
    # eigenvalue = np.array(eigenvalue,dtype=float)
    #do principle component analysis
    new_data_set = PCA(eigenvalue,data_set)
    new_dimension_data = get_new_points(data_set,eigenvector)
    # print (data_set)
    # print (new_dimension_data)
    
    # plt.show()
    # data_set = normalization(data_set)
    # co_variance_matrix = get_C(data_set)
    # eigenvalue,eigenvector = get_eigen(co_variance_matrix)
    # plt.plot(eigenvalue,'.')
    # plt.show()

    data1 = np.random.randn(500, 2)
    for i in range(len(data1)):
        data1[i][0] = data1[i][0]+100
    data2 = np.random.randn(500, 2)-0
    data3 = np.random.randn(500, 2)-100
    data1 = np.append(data1,data2,axis = 0)
    data1 = np.append(data1,data3,axis = 0)
    k_means_clustering(4,data1)
    # plt.plot(data1[:,0],data1[:,1],'+')
    # plt.show()


