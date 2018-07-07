from sklearn.datasets import load_boston
import numpy as np
import scipy.linalg
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import pandas as pd
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
    return np.dot(np.transpose(xis),xis)/len(xis)
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
    data_set = pd.read_csv("C:\\Users\\ZM\\Desktop\\IDA\\2016 School Explorer.csv")
    titles = data_set.columns
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
    data_set = list(data_set.values)
    for i in range(len(data_set)):
        if (type(data_set[i][4])=="float"):
            #数据集的索引4需要转化为float
            data_set[i][4] = float(data_set[i][4])
        else:
            data_set[i][4] = 0.0
        #数据集的索引6需要去掉美元符号和逗号并且转化为float
        if_nan = (data_set[i][6]==data_set[i][6])
        if (if_nan==False):
            data_set[i][6] = 0.0
        else:
            data_set[i][6]= data_set[i][6].replace("$","")
            data_set[i][6]= float(data_set[i][6].replace(",",""))
        #from column 7 to column 18
        for j in range(7,19):
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
if __name__=="__main__":
    
    data_set = data_pruning_for_school_explorer()
    C = get_C(data_set)
    eigenvalue,eigenvector = get_eigen(C)
    new_data_set = PCA(eigenvalue,data_set)
    print(new_data_set.shape)
    plt.show()
    # data_set = normalization(data_set)
    # co_variance_matrix = get_C(data_set)
    # eigenvalue,eigenvector = get_eigen(co_variance_matrix)
    # plt.plot(eigenvalue,'.')
    # plt.show()
    
    
    # data_value = normalization(data_value)
    

