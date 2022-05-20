# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:12:40 2022

@author: yakupcatalkaya
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import time
import random
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

sns.set() # use seaborn plotting style

def pickle_it(files=[],names=None,instruction="wb"):
    if instruction == "wb":
        for idx,file in enumerate(files):
            pickle_out=open(names[idx] + ".pickle", instruction)
            pickle.dump(file,pickle_out)
            pickle_out.close()
    else:
        for name in names:
            pickle_out=open(name + ".pickle", instruction)
            file = pickle.load(pickle_out)
            files.append(file)
            pickle_out.close()
    return files


def similarity(doc1,doc2):
    documents = [doc1,doc2]
    count_vectorizer = CountVectorizer()     
    sparse_matrix = count_vectorizer.fit_transform(documents)
    sparse_matrix = sparse_matrix.astype(np.float32)
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix)
    cosi = cosine_similarity(df,df)
    return cosi[0][1]

def f_score(y_true,y_pred,labell=" "):
    pass

def tester(train,test,def1=" ",def2=" "):
    # global labels,true_label,test_docs,similarity_list,predicted_label,document_two,document_one,testt,trainn
    # testt=test
    # trainn=train
    test_doc_num = sum([len(x) for x in list(test.values())])
    true = 0
    labels = []
    test_docs = []
    predicted_label = []
    key_dic = {"arts":0,"business":1,"politics":2,"sports":3,"technology":4}
    for key in test:
        for document in test[key]:
            test_docs.append([document,key])
    random.shuffle(test_docs)
    true_label = np.array([key_dic[x[1]] for x in test_docs])
    for document in test_docs:
        document_one, key = document
        document_one = " ".join(document_one)
        similarity_list = []
        for aclass in train:
            document_two = " ".join(train[aclass])
            rate = similarity(document_one,document_two)
            similarity_list.append([rate,aclass])
        similarity_list.sort(key=lambda x:x[0],reverse=True)
        label = similarity_list[0][1]
        ratee = similarity_list[0][0]
        predicted_label.append(key_dic[label])
        labels.append(ratee)
        if label == key: true += 1
    accuracy = true / test_doc_num
    
    f1_score(true_label, predicted_label,average="macro")
    print(classification_report(true_label, predicted_label))

    print("Train set : ", def1)
    print("Test set : ", def2)
    print("Accuracy : % ",accuracy*100)
    print("Truly classified number is ", true)
    print("The tested document number is ",str(test_doc_num),"\n")
    print("-"*60+"\n")

def naive(train,test,train_weight,def1=" ",def2=" "):
    global model,X,y,sparse_matrix,test_docs,fature,doc,doc_term_matrix,df,documents,train_set,test_set,y
    key_dic = {"arts":0,"business":1,"politics":2,"sports":3,"technology":4}
    train_features = []
    train_features_weight = []
    test_docs = []
    fature = []
    train_doc = [" ".join(x) for x in train_class.values()]
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(train_doc)
    sparse_matrix = np.array(sparse_matrix.todense(),dtype="float32")
    # df = pd.DataFrame(doc_term_matrix)

    for key in train:
        for feature in train[key]:
            if feature not in fature:fature.append(feature)
    sparse_matrix = np.zeros((5,len(fature)),dtype="float32")
    for key in train:
        for feature in train[key]:
            train_features.append([feature,key_dic[key]])
    for key in train_weight:
         for idx,weight in enumerate(train_weight[key]):       
             train_features_weight.append(weight)
             sparse_matrix[key_dic[key]][fature.index(train_class[key][idx])] = weight
    model = MultinomialNB()
    y = [x[1] for x in train_features]
    model.fit(sparse_matrix, [0,1,2,3,4])
    # weight = np.array([train_features_weight[idx] for idx in range(len(train_features))])
    
    # X = np.array([x[0] for x in train_features]).reshape(-1, 1)
    
    
    for key in test:
        for line in test[key]:
            test_docs.append([line,key_dic[key]])
    

    sys.exit(0)
    sparse_matrix = count_vectorizer.transform([" ".join(doc[0]) for doc in test_docs])
    
    sparse_matrix = np.array(sparse_matrix.todense(),dtype="float32")
    df = pd.DataFrame(sparse_matrix)
    # sys.exit(0)

    predicted = model.predict(df)
    print(predicted)
    
    mat = confusion_matrix([x[1] for x in train_features], predicted)
    sns.heatmap(mat.T, square = True, annot=True, fmt = "d")
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.show()
    true = 0
    true_label = [label[1] for label in test_docs]
    for idx,predict in enumerate(predicted):
        if predict==true_label[idx]:true+=1
    f1_score(true_label, predicted,average="macro")
    print(classification_report(true_label, predicted))
    accuracy=true/len(test_docs)
    print("Train set : ", def1)
    print("Test set : ", def2)
    print("Accuracy : % ",accuracy*100)
    print("Truly classified number is ", true)
    print("The tested document number is ",str(len(test_docs)),"\n")
    print("-"*60+"\n")

def tfidf():
    model = TfidfVectorizer()
    pass
    return model
    
def soft_cos(doc1,doc2):
    documents = [doc1,doc2]
    from gensim.corpora import Dictionary
    from gensim.models import TfidfModel
    import gensim.downloader as api
    from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
    model = api.load('word2vec-google-news-300')
    dictionary = Dictionary(documents)
    doc1 = dictionary.doc2bow(doc1)
    doc2 = dictionary.doc2bow(doc2)
    documents = [doc1,doc2]
    tfidf = TfidfModel(documents)
    doc1 = tfidf[doc1]
    doc2 = tfidf[doc2]
    termsim_index = WordEmbeddingSimilarityIndex(model)
    termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
    similarity = termsim_matrix.inner_product(doc1, doc2, normalized=(True, True))
    return similarity

def merge(dict1,dict2,istest=False):
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = list(value) + list(dict1[key])
               if istest: dict3[key] = [dict3[key][idx] + dict3[key][idx + len(dict3[key])//2] for idx in range(len(dict3[key])//2)]
   return dict3


def main():
    pass


if __name__ == "__main__":    
    start_time = time.time()
    train_class,train_class_weight,test_class,test_sememe,train_sememes,train_sememes_weights,train_unique_sememes,train_unique_sememes_weights = pickle_it(names=["train_class","train_class_weight","test_class","test_sememe","train_sememes","train_sememes_weights","train_unique_sememes","train_unique_sememes_weights"],instruction="rb")

        
    # tester(train=train_class,test=test_class,def1="Corpus",def2="Corpus")
    # tester(train=train_class,test=test_sememe,def1="Corpus",def2="Sememe")
    # tester(train=train_sememes,test=test_class,def1="Sememe",def2="Corpus")
    # tester(train=train_sememes,test=test_sememe,def1="Sememe",def2="Sememe")
    # tester(train=train_unique_sememes,test=test_class,def1="Unique Sememe",def2="Corpus")
    # tester(train=train_unique_sememes,test=test_sememe,def1="Unique Sememe",def2="Sememe")
    # tester(train=merge(train_class,train_sememes),test=test_class,def1="Corpus + Sememe",def2="Corpus")
    # tester(train=merge(train_class,train_sememes),test=test_sememe,def1="Corpus + Sememe",def2="Sememe")
    # tester(train=merge(train_class,train_unique_sememes),test=test_class,def1="Corpus + Unique Sememe",def2="Corpus")
    # tester(train=merge(train_class,train_unique_sememes),test=test_sememe,def1="Corpus + Unique Sememe",def2="Sememe")
    # tester(train=train_class,test=merge(test_class,test_sememe,istest=True),def1="Corpus",def2="Corpus + Sememe")
    # tester(train=train_sememes,test=merge(test_class,test_sememe,istest=True),def1="Sememe",def2="Corpus + Sememe")
    # tester(train=train_unique_sememes,test=merge(test_class,test_sememe,istest=True),def1="Unique Sememe",def2="Corpus + Sememe")
    # tester(train=merge(train_class,train_sememes),test=merge(test_class,test_sememe,istest=True),def1="Corpus + Sememe",def2="Corpus + Sememe")
    # tester(train=merge(train_class,train_unique_sememes),test=merge(test_class,test_sememe,istest=True),def1="Corpus + Unique Sememe",def2="Corpus + Sememe")
    
    
    
    naive(train=train_class,test=test_class,train_weight=train_class_weight,def1="Corpus",def2="Corpus")
    # naive(train=train_class,test=test_sememe,train_weight=train_class_weight,def1="Corpus",def2="Sememe")
    # naive(train=train_sememes,test=test_class,train_weight=train_sememes_weights,def1="Sememe",def2="Corpus")
    # naive(train=train_sememes,test=test_sememe,train_weight=train_sememes_weights,def1="Sememe",def2="Sememe")
    # naive(train=train_unique_sememes,test=test_class,train_weight=train_unique_sememes_weights,def1="Unique Sememe",def2="Corpus")
    # naive(train=train_unique_sememes,test=test_sememe,train_weight=train_unique_sememes_weights,def1="Unique Sememe",def2="Sememe")
    # naive(train=merge(train_class,train_sememes),test=test_class,train_weight=merge(train_class_weight,train_sememes_weights),def1="Corpus + Sememe",def2="Corpus")
    # naive(train=merge(train_class,train_sememes),test=test_sememe,train_weight=merge(train_class_weight,train_sememes_weights),def1="Corpus + Sememe",def2="Sememe")
    # naive(train=merge(train_class,train_unique_sememes),test=test_class,train_weight=merge(train_class_weight,train_unique_sememes_weights),def1="Corpus + Unique Sememe",def2="Corpus")
    # naive(train=merge(train_class,train_unique_sememes),test=test_sememe,train_weight=merge(train_class_weight,train_unique_sememes_weights),def1="Corpus + Unique Sememe",def2="Sememe")
    # naive(train=train_class,test=merge(test_class,test_sememe,istest=True),train_weight=train_class_weight,def1="Corpus",def2="Corpus + Sememe")
    # naive(train=train_sememes,test=merge(test_class,test_sememe,istest=True),train_weight=train_sememes_weights,def1="Sememe",def2="Corpus + Sememe")
    # naive(train=train_unique_sememes,test=merge(test_class,test_sememe,istest=True),train_weight=train_unique_sememes_weights,def1="Unique Sememe",def2="Corpus + Sememe")
    # naive(train=merge(train_class,train_sememes),test=merge(test_class,test_sememe,istest=True),train_weight=merge(train_class_weight,train_sememes_weights),def1="Corpus + Sememe",def2="Corpus + Sememe")
    # naive(train=merge(train_class,train_unique_sememes),test=merge(test_class,test_sememe,istest=True),train_weight=merge(train_class_weight,train_unique_sememes_weights),def1="Corpus + Unique Sememe",def2="Corpus + Sememe")

    print("The time spent is", str(int(time.time()-start_time))," seconds.")
    main()
            
    


    
    
