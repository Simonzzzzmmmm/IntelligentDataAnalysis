from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
import random
import sys
sys.path.append("C:\IntelligentDataAnalysis\Principle_Component_Analysis")

import numpy as np
import math
import re
def PCA(eigenvalue,eigenvector,xis):
    total_value = 0
    for i in range(len(eigenvalue)):
        total_value+=eigenvalue[i]
    ratio = 0
    total_real_value = 0
    new_xis = []
    rows = 0
    new_eigenvalue = []

    while (ratio != 0.9):
        max_index = np.argmax(eigenvalue)
        new_xis.append(xis[:,max_index])
        #delete the dimension that has the max value, for the following argmax.
        xis = np.delete(xis,max_index,axis=1)
        new_eigenvalue.append(eigenvalue[max_index])
        eigenvalue = np.delete(eigenvalue,eigenvalue[max_index],axis=0)
        total_real_value+=eigenvalue[max_index]
        rows+=1
        if (total_real_value/total_value>=0.8):
            break
    # new_eigenvector = np.zeros((rows,rows))
    new_eigenvector = eigenvector[:rows,:rows]
    new_axis = np.transpose(np.array(new_xis))
    return new_axis,new_eigenvector,np.array(new_eigenvalue)
def go_split(s, symbol):
    # 拼接正则表达式
    symbol = "[" + symbol + "]+"
    # 一次性分割字符串
    result = re.split(symbol, s)
    # 去除空字符
    return [x for x in result if x]
def get_document(file_path,symbol):
    frequency = {}
    document = []
    symbol = "(), \n."
    for line in open(file_path).readlines():
        line = go_split(line, symbol)
        if (len(line) != 0):
            document.extend(line)
    return document
def get_fre(document):
    frequency = {}
    for i in range(len(document)):
        if ((document[i] in frequency.keys())==True):
            frequency[document[i]]+=1/len(document)
        else:
            frequency[document[i]]=1/len(document)
    return frequency
def doc_vector():
    file_path1 = "report1.txt"
    file_path2 = "report2.txt"
    file_path3 = "report3.txt"
    symbol = "(), \n."
    document1 = get_document(file_path1,symbol)
    frequency1 = get_fre(document1)
    document2 = get_document(file_path2, symbol)
    frequency2 = get_fre(document2)
    document3 = get_document(file_path3, symbol)
    frequency3 = get_fre(document3)

    all_word = []
    all_word.extend(frequency1.keys())
    all_word.extend(frequency2.keys())
    all_word.extend(frequency3.keys())

    #tfidf
    Nk = {}
    for word in range(len(all_word)):
        if ((all_word[word] in Nk.keys())==True):
            Nk[all_word[word]]+=1
        else:
            Nk[all_word[word]] =1
    #calculate xik
    for key in frequency1.keys():
        frequency1[key] = frequency1[key]*math.log2(3/Nk[key])
    for key in frequency2.keys():
        frequency2[key] = frequency2[key]*math.log2(3/Nk[key])
    for key in frequency3.keys():
        frequency3[key] = frequency3[key]*math.log2(3/Nk[key])
    return frequency1,frequency2,frequency3
def get_C(vector):
    return np.dot(np.transpose(vector),vector)
def eigen(C):
    return np.linalg.eig(C)
def doc_sequence():
    pass

if __name__ == "__main__":
    frequency1, frequency2, frequency3 = doc_vector()
    C1 = get_C(frequency1)
    C2 = get_C(frequency2)
    C3 = get_C(frequency3)

    print(sys.path)