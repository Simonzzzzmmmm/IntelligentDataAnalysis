import numpy as np
from sklearn.preprocessing import Imputer
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
def mean_preprocessing(data_set):
    imp = Imputer(missing_values='NaN',strategy='mean',axis=1)
    return imp.transform(data_set)
def label_for_school_explorer(dataset):
    pass
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
    mean_money = np.mean(data_set[:,5])
    # label it with high, median, low
    for i in range(len(data_set[:,5])):
        if (data_set[i][5]>60000):
            data_set[i][5]=1
        elif (data_set[i][5]>40000):
            data_set[i][5]=0
        else:
            data_set[i][5]=-1
    return data_set