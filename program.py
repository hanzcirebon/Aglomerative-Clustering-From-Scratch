import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import sys 

def NaiveBayes(X_train, X_test, Y_train, Y_test):
    # Fitting classifier to the Training set
    model = GaussianNB().fit(X_train,Y_train)

    # Predicting the Test set results
    Y_pred = model.predict(X_test)

    # getting the accuracy
    accuracy = accuracy_score(Y_test,Y_pred)
    print("Naive Bayes Classification Accuracy : {}".format(accuracy))

def KNN(X_train, X_test, Y_train, Y_test):
    # Fitting classifier to the Training set
    model = KNeighborsClassifier(n_neighbors=100).fit(X_train,Y_train)

    # Predicting the Test set results
    Y_pred = model.predict(X_test)

    # getting the accuracy
    accuracy = accuracy_score(Y_test,Y_pred)
    print("KNearestN Classification Accuracy : {}\n".format(accuracy))

def Cluster_Agglomerative(X, no_cluster):
    # creating data for clustering
    x = X.iloc[:,0]
    y = X.iloc[:,1]
    euclidian = []

    # creating array of euclidian
    for i in range(len(x)):
        for j in range(i+1,len(y)):
            # finding distance between point
            a = abs(x[i]-x[j])
            b = abs(y[i]-y[j])
            c = math.sqrt(a*a + b*b)
    
            # append distance from 2 points into euclidian array
            euclidian.append({'p_1':i,'p_2':j,'dst':c})

    # clustering start
    n_column = ['cluster','point']
    cluster = pd.DataFrame(columns = n_column)
    no = 1
    idx = 0
    while True:

        # if euclidian list is empty end looping
        if not euclidian:
            break
        else:
            # find minimum distance
            min_data = min(euclidian, key=lambda x:x['dst'])

            # find index of min_data
            min_idx = euclidian.index(min_data)
            p_1, p_2 = min_data['p_1'], min_data['p_2']
            p1, p2 = int(p_1), int(p_2)

            # insert minimum distance point into cluster
            cluster.loc[idx] = [no,[p1,p2]]
            idx += 1
            no += 1

            # delete the minimum distance from euclidian list
            euclidian.pop(min_idx)

            # find list of data p1 and p2
            l_p1 = list(filter(lambda p: (p['p_2'] == p1 or p['p_1'] == p1), euclidian))
            l_p2 = list(filter(lambda p: (p['p_2'] == p2 or p['p_1'] == p2), euclidian))

            # finding max distance between target of cluster -no
            for j in range(len(l_p1)):

                # both target are at p1
                if l_p1[j]['p_1'] == l_p2[j]['p_1']:
                    # if distance target p2 >= p1 then replace distance p1 with p2, otherwise skip
                    if l_p1[j]['dst'] >= l_p2[j]['dst']:
                        pass
                    else:
                        l_p1[j]['dst'] = l_p2[j]['dst']

                # both target are at p2
                elif l_p1[j]['p_2'] == l_p2[j]['p_2']:
                    # if distance target p2 >= p1 then replace distance p1 with p2, otherwise skip
                    if l_p1[j]['dst'] >= l_p2[j]['dst']:
                        pass
                    else:
                        l_p1[j]['dst'] = l_p2[j]['dst']

                # both target are at p1 and p2
                elif l_p1[j]['p_1'] == l_p2[j]['p_2']:
                    # if distance target p2 >= p1 then replace distance p1 with p2, otherwise skip
                    if l_p1[j]['dst'] >= l_p2[j]['dst']:
                        pass
                    else:
                        l_p1[j]['dst'] = l_p2[j]['dst']

                # both target are at p2 and p1
                elif l_p1[j]['p_2'] == l_p2[j]['p_1']:
                    # if distance target p2 >= p1 then replace distance p1 with p2, otherwise skip
                    if l_p1[j]['dst'] >= l_p2[j]['dst']:
                        pass
                    else:
                        l_p1[j]['dst'] = l_p2[j]['dst']
                
                # remove unused data
                euclidian.remove(l_p2[j])
    
    # separate cluster into 8 cluster
    cluster.drop(cluster.tail(no_cluster-1).index,inplace=True)
    i = 1
    new_cluster = []
    while i <= no_cluster:

        # get the last cluster data inserted into arr
        arr = cluster.iloc[-1]['point']

        # delete the last cluster data from cluster dataset
        data = cluster.iloc[-1]
        cluster = cluster[cluster.cluster != data.cluster]

        # get last index of cluster
        j = len(cluster) -1
        while j >= 0:

            # check if the point is exist in arr or not
            if (cluster.iloc[j]['point'][0] in arr or cluster.iloc[j]['point'][1] in arr):

                # if the point[0] is exist in array, then point[1] inserted in arr
                if cluster.iloc[j]['point'][0] in arr:
                    arr.extend([cluster.iloc[j]['point'][1]])

                # if the point[1] is exist in array, then point[0] inserted in arr
                else:
                    arr.extend([cluster.iloc[j]['point'][0]])

                # delete the array that has been used
                data = cluster.iloc[j]
                cluster = cluster[cluster.cluster != data.cluster]

            else:
                pass

            j -=1

        # insert the arr into new cluster
        new_cluster.append(arr)
        i +=1
    
    for i in range(1,len(new_cluster)+1):
        print ('cluster ',i,' = ',new_cluster[i-1])

def Encode_Scale_Split(X,Y):
    # encode data using label encoder
    le = LabelEncoder()
    X = X.apply(le.fit_transform)
    Y = le.fit_transform(Y)

    # feature scaling 
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, Y_train, Y_test = tts(X, Y, test_size = 0.25, random_state = 0)

    return X_train, X_test, Y_train, Y_test

def Data_preparation_1(data):
    # droping rows that has an empty column
    df = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    
    df['availability_365'] = np.where(df['availability_365'] > 0, 'availabe', df['availability_365'])
    df['availability_365'] = np.where(df['availability_365'] != 'availabe', 'full', df['availability_365'])

    # selecting used column
    df = df.iloc[:,np.r_[4:6,8:16]]

    #df.to_csv ('data_classification_1.csv', index = False, header=True)

    # selecting feature and label data
    X = df.iloc[:,np.r_[0:9]]
    Y = df.iloc[:,9]

    return X,Y

def Data_preparation_2(data):
    # droping rows that has an empty column
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    
    # selecting used column
    data = data.iloc[:,np.r_[8:12]]

    #data.to_csv ('data_classification_2.csv', index = False, header=True)

    # selecting feature and label data
    X = data.iloc[:,np.r_[1:4]]
    Y = data.iloc[:,0]

    return X,Y

def Data_preparation_cluster(data):
    # droping rows that has an empty column
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # finding the mean of data that grouped by Neighbors column
    data = data.groupby(['neighbourhood']).mean()

    # getting neighborhood latitude and longitude mean
    X1 = data.iloc[:,np.r_[2,3]]
    X2 = data.iloc[:,np.r_[4,6]]

    #X1.to_csv ('data_cluster_1.csv', index = False, header=True)
    #X2.to_csv ('data_cluster_2.csv', index = False, header=True)

    return X1, X2
    

def main():
    # read dataset
    data = pd.read_csv('air_bnb.csv')
    
    #################
    # classification
    #################

    # eksperimen 1 dan 2
    print ("Experiment 1 \n")
    X,Y = Data_preparation_1(data)
    X_train, X_test, Y_train, Y_test = Encode_Scale_Split(X,Y)
    NaiveBayes(X_train, X_test, Y_train, Y_test)
    KNN(X_train, X_test, Y_train, Y_test)
    
    # eksperimen 3 dan 4
    print ("Experiment 2 \n")
    X,Y = Data_preparation_2(data)
    X_train, X_test, Y_train, Y_test = Encode_Scale_Split(X,Y)
    NaiveBayes(X_train, X_test, Y_train, Y_test)
    KNN(X_train, X_test, Y_train, Y_test)

    #################
    # clustering
    #################

    # preparing data for X1 and X2, X1 will go to eksperimen 5 and X2 will go to eksperimen 6
    print ("Agglomerative Complete Link Maximum cluster\n")
    X1,X2 = Data_preparation_cluster(data)

    # eksperimen 5
    print("##############")
    print('data Cluster 1')
    print(X1)
    print ("Clustering latitude and longitude to find which neighbourhood stays in the same neighbourhood_Group\n")
    Cluster_Agglomerative(X1,12)
    print ("\n")

    # eksperimen 6
    print("##############")
    print('data Cluster 2')
    print(X2)
    print ("Clustering price and number of review to find which neighbourhood is worth to stay\n")
    Cluster_Agglomerative(X2,3)

if __name__ == '__main__':
    main()