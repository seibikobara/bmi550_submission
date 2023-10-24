
'''
Assignment 2
Seibi Kobara
'''





import pandas as pd
import numpy as np

import re
import sys


# NLP
from nltk import word_tokenize
from nltk.stem.porter import *
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
import codecs
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec

# model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# figures
import matplotlib.pyplot as plt

# tf
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from math import nan
from keras.callbacks import ModelCheckpoint
from subprocess import check_output
import tensorflow as tf


# function 
def loadwordclusters():
    word_clusters = {}
    infile = open('/Users/seibi/projects/bmi550/wk8/50mpaths2')
    for line in infile:
        items = str.strip(line).split()
        class_ = items[0]
        term = items[1]
        word_clusters[term] = class_
    return word_clusters


def getclusterfeatures(sent):
    sent = sent.lower()
    terms = word_tokenize(sent)
    cluster_string = ''
    for t in terms:
        if t in word_clusters.keys():
                cluster_string += 'clust_' + word_clusters[t] + '_clust '
    return str.strip(cluster_string)


def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df


def preprocess_text(raw_text):
    '''
        Preprocessing function
        PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the
        preprocessed string after applying a series of functions.
    '''
    # Replace/remove username
    # raw_text = re.sub('(@[A-Za-z0-9\_]+)', '@username_', raw_text)
    #stemming and lowercasing (no stopword removal
    words = [stemmer.stem(w) for w in raw_text.lower().split()]
    return (" ".join(words))



# load training set
path = "/Users/seibi/projects/bmi550/assignment2/fallreports_2023-9-21_train.csv"
data = pd.read_csv("/Users/seibi/projects/bmi550/assignment2/fallreports_2023-9-21_train.csv")


# missing values in fall_description
# missing imputation may not feasible. Assumeing this is MCAR, total case analysis.
data = data.dropna(subset = "fall_description")
data = data.reset_index(drop = True)
data.shape


texts = data["fall_description"]
preprocessed_text = [preprocess_text(t) for t in texts ]

classes = data["fog_q_class"]


# demogramhics preparation
temp = data.loc[:,["education", "gender"]]
enc = OneHotEncoder(handle_unknown = "ignore")
demos = enc.fit_transform(temp).toarray()


# incidence location 
temp = data.loc[:,["fall_location"]]
enc = OneHotEncoder(handle_unknown = "ignore")
fall_locs = enc.fit_transform(temp).toarray()




#---------------------------------
# machine learning architecture
#---------------------------------
# SVM
pipe_svm = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", svm.SVC())])

params_svm = {
        "classifier__kernel":["linear","rbf","sigmoid"],
        "classifier__C":[1,5,10]
        }


# decision tree
pipe_dt = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", tree.DecisionTreeClassifier())])

params_dt = {}


# XGB
pipe_xgb = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", XGBClassifier())])


params_xgb = {
        "classifier__max_depth":[6,10],
        "classifier__n_estimators":[10,50]
        }

# logictic regression
pipe_lg = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", LogisticRegression())])

params_lg = {}


# KNN
pipe_knn = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", KNeighborsClassifier())])

params_knn = {
        "classifier__n_neighbors":[5,10,50],
        "classifier__leaf_size": [30, 50]
        }

# NB
pipe_nb =  Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", GaussianNB())])

params_nb = {}


classifiers = ["SVM", "Decision tree", "XGBoost","Logistic regression", "KNN", "Naive Bayes",]
pipes = [pipe_svm, pipe_dt, pipe_xgb, pipe_lg, pipe_knn, pipe_nb]
params = [params_svm, params_dt, params_xgb, params_lg, params_knn,params_nb]



cv_results = {}
# cross validation
skf = StratifiedKFold(n_splits= 3, shuffle=False)
cv_time = 1
for train_index, test_index in skf.split(preprocessed_text, classes):
    
    train_cv_x = pd.Series(preprocessed_text)[train_index].tolist()
    train_cv_y = pd.Series(classes)[train_index].tolist()

    test_cv_x = pd.Series(preprocessed_text)[test_index].tolist()
    test_cv_y = pd.Series(classes)[test_index].tolist()


    # ngram (feature set 1)
    vectorizer = CountVectorizer(ngram_range=(1,3), analyzer="word", tokenizer=None, preprocessor=None, max_features=1000000)

    train_cv_x_ngram_mat = vectorizer.fit_transform(train_cv_x).toarray()
    test_cv_x_ngram_mat = vectorizer.transform(test_cv_x).toarray()
    
    


    # word cluster (feature set 2)
    word_clusters = loadwordclusters()

    training_cv_x_cluster = []
    for i in train_cv_x:
        training_cv_x_cluster.append(getclusterfeatures(i))
    test_cv_x_cluster = []
    for j in test_cv_x:
        test_cv_x_cluster.append(getclusterfeatures(j))

    vectorizer = CountVectorizer(ngram_range=(1,1), analyzer="word", tokenizer=None, preprocessor=None, max_features=100000)

    train_cv_x_cluster_mat = vectorizer.fit_transform(training_cv_x_cluster ).toarray()
    test_cv_x_cluster_mat = vectorizer.transform(test_cv_x_cluster).toarray()


    # word2vec (feature set 3)
    # entiry has to be word levels for each data entry
    train_cv_x_wordsplit = [i.split() for i in train_cv_x]
    w2v_model = Word2Vec(train_cv_x_wordsplit, vector_size=100, window = 5, min_count = 2)

    # terms
    words_in_w2v = w2v_model.wv.index_to_key

    # apply each dense vector to each sentence
    train_cv_x_w2v = []
    for t in train_cv_x_wordsplit:
        # print(t)
        # for each text entry
        temp = []
        for w in t:
            if w in words_in_w2v:
                temp.append(w2v_model.wv[w])
            else:
                temp.append(np.zeros((100)))
                # aggregate by taking average, while keeping vector dimensions
    #  print(np.array(temp).shape)
        temp1= np.mean(np.array(temp), axis = 0).tolist()
        #print(np.array(temp1).shape)
        #print(temp1)
        train_cv_x_w2v.append(temp1)

    # apply the same density vector for test
    test_cv_x_wordsplit = [i.split() for i in test_cv_x]
    test_cv_x_w2v = []
    for t in test_cv_x_wordsplit:
        # print(t)
        # for each text entry
        temp = []
        for w in t:
            if w in words_in_w2v:
                temp.append(w2v_model.wv[w])
            else:
                temp.append(np.zeros((100)))
                # aggregate by taking average, while keeping vector dimensions
        # print(np.array(temp).shape)
        temp1= np.mean(np.array(temp), axis = 0).tolist()
        #print(np.array(temp1).shape)
        #print(temp1)
        test_cv_x_w2v.append(temp1)

    train_cv_x_w2v_mat = np.array(train_cv_x_w2v)
    test_cv_x_w2v_mat = np.array(test_cv_x_w2v)



    # demographics (feature set 4)
    train_cv_x_demo_mat = demos[train_index,:]
    test_cv_x_demo_mat = demos[test_index,:]


    # fall incidence location (feature set 5)
    train_cv_x_fall_mat = fall_locs[train_index,:]
    test_cv_x_fall_mat = fall_locs[test_index,:]


    feats_desc =["ngram","cluster","word2vec","demographics","fall_location"]
    train_feats_list = [train_cv_x_ngram_mat, train_cv_x_cluster_mat, train_cv_x_w2v_mat, train_cv_x_demo_mat, train_cv_x_fall_mat]
    test_feats_list = [test_cv_x_ngram_mat, test_cv_x_cluster_mat, test_cv_x_w2v_mat, test_cv_x_demo_mat, test_cv_x_fall_mat]

    for feat in range(len(feats_desc)):
        feat_ = feats_desc[feat]
        train_cv_selected = train_feats_list[feat]
        test_cv_selected = test_feats_list[feat]


        for i in range(len(pipes)):
            cls = classifiers[i]
            pipe_ = pipes[i]
            params_ = params[i]
            grid = GridSearchCV(estimator=pipe_,
                                param_grid=params_,
                                refit = True,
                                cv = 2,
                                return_train_score=False,
                                scoring = "f1_micro")

            grid.fit(train_cv_selected, train_cv_y)

            # test in the cv
            y_pred = grid.predict(test_cv_selected)
            y_true = test_cv_y 
            # metric
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_macro = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)

            # record
            result = pd.DataFrame(grid.cv_results_)
            n , m = result.shape
            result["feature"] = np.repeat(feat_, n)
            result["classifier"] = np.repeat(cls, n)
            result["F1_micro_in_cv_test"] = np.repeat(f1_micro, n)
            result["F1_macro_in_cv_test"] = np.repeat(f1_macro, n)
            result["accuracy_in_cv_test"] = np.repeat(accuracy, n)

            temp_ = str(cv_time) + "-" + feat_ + "-" + cls
            cv_results[temp_] = result


            # result output
            print("CV round: {:d}".format(cv_time))
            print("Feature: " +feat_)
            print("Classifier: " + cls)


    cv_time += 1



# summary
res = pd.concat(cv_results.values(), axis = 0)


# best model feature combination by F1_micro
best_model_in_eachCV = res[res["rank_test_score"]==1]
best_model_in_eachCV.to_csv("best_model.csv")

# report best classifier and feature set
max_value = best_model_in_eachCV["F1_micro_in_cv_test"].max()
temp = best_model_in_eachCV[best_model_in_eachCV["F1_micro_in_cv_test"]==max_value]
temp.loc[:,["feature","classifier"]]


# obtain the parameters
best_model_in_eachCV[(best_model_in_eachCV["F1_micro_in_cv_test"]==max_value) & (best_model_in_eachCV["classifier"]=="SVM")]
# C;5, kernel = sigmoid 





# ---------------------------------
# sample size required to achieve the highest performance
#----------------------------------

# track information
sizes = []
f1s = []

pipe_svm = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", svm.SVC())])

params_svm = {
        "classifier__kernel":["sigmoid"],
        "classifier__C":[5]
        }

# different sizes were tested using different split threshold
for split in range(2,10,1):
    skf = StratifiedKFold(n_splits= split, shuffle=False)
    for train_index, test_index in skf.split(preprocessed_text, classes):

        train_cv_x = pd.Series(preprocessed_text)[train_index].tolist()
        train_cv_y = pd.Series(classes)[train_index].tolist()

        test_cv_x = pd.Series(preprocessed_text)[test_index].tolist()
        test_cv_y = pd.Series(classes)[test_index].tolist()


        # size of taining set
        size = len(train_cv_x)

        # ngram (feature set 1)
        vectorizer = CountVectorizer(ngram_range=(1,3), analyzer="word", tokenizer=None, preprocessor=None, max_features=1000000)

        train_cv_x_ngram_mat = vectorizer.fit_transform(train_cv_x).toarray()
        test_cv_x_ngram_mat = vectorizer.transform(test_cv_x).toarray()


        grid = GridSearchCV(estimator=pipe_,
                            param_grid=params_,
                            refit = True,
                            cv = 2,
                            return_train_score=False,
                            scoring = "f1_micro")
        grid.fit(train_cv_x_ngram_mat, train_cv_y)
        # test in the cv
        y_pred = grid.predict(test_cv_x_ngram_mat)
        y_true = test_cv_y 
        # metric
        f1_micro = f1_score(y_true, y_pred, average='micro')
    
        f1s.append(f1_micro)
        sizes.append(size)


sample_res = pd.DataFrame({
    "size":sizes,
    "f1":f1s
})
sample_res.to_csv("sample_size.csv")



#-----------------------------------
# Ablation study
#-----------------------------------

# Use all feature sets and remove one at a time
# SVM
pipe_svm = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", svm.SVC())])

params_svm = {
        "classifier__kernel":["sigmoid"],
        "classifier__C":[5]
        }


# for record
removes = []
f1_micros = []
f1_macros = []
accuracys = []


# cross validation
skf = StratifiedKFold(n_splits= 3, shuffle=False)
cv_time = 1
for train_index, test_index in skf.split(preprocessed_text, classes):
    
    train_cv_x = pd.Series(preprocessed_text)[train_index].tolist()
    train_cv_y = pd.Series(classes)[train_index].tolist()

    test_cv_x = pd.Series(preprocessed_text)[test_index].tolist()
    test_cv_y = pd.Series(classes)[test_index].tolist()


    # ngram (feature set 1)
    vectorizer = CountVectorizer(ngram_range=(1,3), analyzer="word", tokenizer=None, preprocessor=None, max_features=1000000)

    train_cv_x_ngram_mat = vectorizer.fit_transform(train_cv_x).toarray()
    test_cv_x_ngram_mat = vectorizer.transform(test_cv_x).toarray()
    
    


    # word cluster (feature set 2)
    word_clusters = loadwordclusters()

    training_cv_x_cluster = []
    for i in train_cv_x:
        training_cv_x_cluster.append(getclusterfeatures(i))
    test_cv_x_cluster = []
    for j in test_cv_x:
        test_cv_x_cluster.append(getclusterfeatures(j))

    vectorizer = CountVectorizer(ngram_range=(1,1), analyzer="word", tokenizer=None, preprocessor=None, max_features=100000)

    train_cv_x_cluster_mat = vectorizer.fit_transform(training_cv_x_cluster ).toarray()
    test_cv_x_cluster_mat = vectorizer.transform(test_cv_x_cluster).toarray()


    # word2vec (feature set 3)
    # entiry has to be word levels for each data entry
    train_cv_x_wordsplit = [i.split() for i in train_cv_x]
    w2v_model = Word2Vec(train_cv_x_wordsplit, vector_size=100, window = 5, min_count = 2)

    # terms
    words_in_w2v = w2v_model.wv.index_to_key

    # apply each dense vector to each sentence
    train_cv_x_w2v = []
    for t in train_cv_x_wordsplit:
        # print(t)
        # for each text entry
        temp = []
        for w in t:
            if w in words_in_w2v:
                temp.append(w2v_model.wv[w])
            else:
                temp.append(np.zeros((100)))
                # aggregate by taking average, while keeping vector dimensions
    #  print(np.array(temp).shape)
        temp1= np.mean(np.array(temp), axis = 0).tolist()
        #print(np.array(temp1).shape)
        #print(temp1)
        train_cv_x_w2v.append(temp1)

    # apply the same density vector for test
    test_cv_x_wordsplit = [i.split() for i in test_cv_x]
    test_cv_x_w2v = []
    for t in test_cv_x_wordsplit:
        # print(t)
        # for each text entry
        temp = []
        for w in t:
            if w in words_in_w2v:
                temp.append(w2v_model.wv[w])
            else:
                temp.append(np.zeros((100)))
                # aggregate by taking average, while keeping vector dimensions
        # print(np.array(temp).shape)
        temp1= np.mean(np.array(temp), axis = 0).tolist()
        #print(np.array(temp1).shape)
        #print(temp1)
        test_cv_x_w2v.append(temp1)

    train_cv_x_w2v_mat = np.array(train_cv_x_w2v)
    test_cv_x_w2v_mat = np.array(test_cv_x_w2v)



    # demographics (feature set 4)
    train_cv_x_demo_mat = demos[train_index,:]
    test_cv_x_demo_mat = demos[test_index,:]


    # fall incidence location (feature set 5)
    train_cv_x_fall_mat = fall_locs[train_index,:]
    test_cv_x_fall_mat = fall_locs[test_index,:]


    feats_desc =["ngram","cluster","word2vec","demographics","fall_location"]
    train_feats_list = [train_cv_x_ngram_mat, train_cv_x_cluster_mat, train_cv_x_w2v_mat, train_cv_x_demo_mat, train_cv_x_fall_mat]
    test_feats_list = [test_cv_x_ngram_mat, test_cv_x_cluster_mat, test_cv_x_w2v_mat, test_cv_x_demo_mat, test_cv_x_fall_mat]

    # remove one
    for remove_idx in range(len(feats_desc)):
        remove = feats_desc[remove_idx]

        # create training matrix without this feature set
        temp = []
        for j in range(len(feats_desc)):
            if not j==remove_idx:
                temp.append(train_feats_list[j])
        train_cv_selected = np.concatenate(temp, axis=1)

        # create test matrix without this feature set
        temp = []
        for j in range(len(feats_desc)):
            if not j==remove_idx:
                temp.append(test_feats_list[j])
        test_cv_selected = np.concatenate(temp, axis=1)



        grid = GridSearchCV(estimator=pipe_svm ,
                            param_grid=params_svm,
                            refit = True,
                            cv = 2,
                            return_train_score=False,
                            scoring = "f1_micro")
        grid.fit(train_cv_selected, train_cv_y)
        # test in the cv
        y_pred = grid.predict(test_cv_selected)
        y_true = test_cv_y 
        
        # metric
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)
        
        # record
        removes.append(remove)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)
        accuracys.append(accuracy)


pd.DataFrame({
    "removed":removes,
    "f1_micro":f1_macros,
    "f1_macro":f1_macros,
    "accuracy":accuracys
}).to_csv("ablation.csv")



#------------------------------------
# final model evaluation
# ------------------------------------
# training set
train_x = preprocessed_text
train_y = classes


# test
path = "/Users/seibi/projects/bmi550/assignment2/fallreports_2023-9-21_test.csv"
test_data = pd.read_csv(path)

# remove NA
test_data = test_data.dropna(subset = "fall_description")
test_data = test_data.reset_index(drop = True)

texts= test_data["fall_description"]
test_x = [preprocess_text(t) for t in texts ]
test_y = test_data["fog_q_class"].tolist()


# track information
sizes = []
f1s = []

pipe_svm = Pipeline([
    ('standard_scaler', StandardScaler()), 
    ("classifier", svm.SVC(kernel="sigmoid", C = 5))])

# bootstrapping
f1s = []
n  = len(train_x)
iters = 100
for i in range(iters):
    idx = np.random.choice(range(n), size = int(n*0.8), replace = True)
    
    train_boot_x = pd.Series(train_x)[idx].tolist()
    train_boot_y = pd.Series(train_y)[idx].tolist()

    # ngram (feature set 1)
    vectorizer = CountVectorizer(ngram_range=(1,3), analyzer="word", tokenizer=None, preprocessor=None, max_features=1000000)

    train_boot_x_ngram_mat = vectorizer.fit_transform(train_boot_x).toarray()

    test_x_ngram_mat = vectorizer.transform(test_x).toarray()

    pipe_svm.fit(train_boot_x_ngram_mat, train_boot_y)

    y_pred = pipe_svm.predict(test_x_ngram_mat)
    y_true = test_y 

    # metric
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1s.append(f1_micro)
    print("Currently: {iteration} %".format(iteration = np.round(i/iters * 100, 2)))


f1 = np.array(f1s)
mean = np.mean(f1)
upper = np.quantile(f1, 0.975)
lower = np.quantile(f1, 0.0250)
print("Mean: {:.2f}, 95%CI:[{:.2f}, {:.2f}]".format(mean, upper, lower))