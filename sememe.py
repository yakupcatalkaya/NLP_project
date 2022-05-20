# -*- coding: utf-8 -*-
"""
Created on Fri May  6 17:12:40 2022

@author: yakupcatalkaya
"""

import numpy as np
import pandas as pd
import OpenHowNet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import time
from nltk import pos_tag
import pickle
import sys


def removal(words):
    dict_pos_map = {'NOUN': "n",'VERB':"v",'ADJ' : "a",'ADV':"r"}
    new_words = []
    for word in words:
        if word[1] in list(dict_pos_map.keys()):
            new_words.append(word)
    return new_words


def sememe_generation(tokens_list,hownet_dict,test_case=False):
    tag_index = 1
    if test_case: tag_index = 2
    se = []
    we = []
    s = []
    w = []
    for tokens in tokens_list:
        sememes = [] 
        count_vectorizer = CountVectorizer()
        tag_dict = {'n':'noun',
                    'a':'adj',
                    'v':'verb',
                    'r':'adv'}
        for token in tokens:
            meanings = hownet_dict.get_sense(token[0],language="en", pos=tag_dict[token[tag_index]]) 
            token_semem_list = list(set([meaning.get_sememe_list()[0].en for meaning in meanings]))
            sememes.append(bytes(" ".join(token_semem_list),"utf-8"))
        sparse_matrix = count_vectorizer.fit_transform(sememes)
        sparse_matrix = sparse_matrix.astype(np.float32)
        doc_term_matrix = sparse_matrix.todense().T
        df = pd.DataFrame(doc_term_matrix)
        sememe_list = count_vectorizer.get_feature_names()
        cosi = cosine_similarity(df,df)
        weight = np.sum(cosi,axis=1)
        sememe_indices = list(np.argsort(weight))
        semem = [sememe_list[x] for x in sememe_indices]
        se.append(semem)
        we.append(weight)
    if not test_case:    
        s.append([x for x in se[0][:100] if x not in se[1][:50] and x not in se[2][:50] and x not in se[3][:50] and x not in se[4][:50]])
        s.append([x for x in se[1][:100] if x not in se[0][:50] and x not in se[2][:50] and x not in se[3][:50] and x not in se[4][:50]])
        s.append([x for x in se[2][:100] if x not in se[1][:50] and x not in se[0][:50] and x not in se[3][:50] and x not in se[4][:50]])
        s.append([x for x in se[3][:100] if x not in se[1][:50] and x not in se[2][:50] and x not in se[0][:50] and x not in se[4][:50]])
        s.append([x for x in se[4][:100] if x not in se[1][:50] and x not in se[2][:50] and x not in se[3][:50] and x not in se[0][:50]])
        
        s_idx = [[],[],[],[],[]]
        for idx,ss in enumerate(s):
            s_idx[idx] = [index for index,x in enumerate(ss)]
        
        for idx,ww in enumerate(s_idx):
            w.append([we[idx][www] for www in ww])
            
        
        return se,we,s,w
    
    return se,we
    

def generator(busines,art,politic,sport,technolog):
    global cosi,df
    arts = " ".join(["".join((art[0]+" ")*int(art[1])) for art in art])
    business = " ".join(["".join((art[0]+" ")*int(art[1])) for art in busines])
    politics = " ".join(["".join((art[0]+" ")*int(art[1])) for art in politic])
    sports = " ".join(["".join((art[0]+" ")*int(art[1])) for art in sport])
    technology = " ".join(["".join((art[0]+" ")*int(art[1])) for art in technolog])
    
    classes = [arts,business,politics,sports,technology]
    count_vectorizer = CountVectorizer()     
    sparse_matrix = count_vectorizer.fit_transform(classes)
    sparse_matrix = sparse_matrix.astype(np.float32)
    doc_term_matrix = sparse_matrix.todense().T
    df = pd.DataFrame(doc_term_matrix)
    tokens = count_vectorizer.get_feature_names()
    cosi = cosine_similarity(df,df)
    weight = np.sum(cosi,axis=1)
    
    art_max_indx = list(np.argsort(df[0])[:-500:-1])
    art_min_indx = list(np.argsort(df[0])[:500])
    business_max_indx = list(np.argsort(df[1])[:-500:-1])
    business_min_indx = list(np.argsort(df[1])[:500])
    politics_max_indx = list(np.argsort(df[2])[:-500:-1])
    politics_min_indx = list(np.argsort(df[2])[:500])
    sports_max_indx = list(np.argsort(df[3])[:-500:-1])
    sports_min_indx = list(np.argsort(df[3])[:500])
    technology_max_indx = list(np.argsort(df[4])[:-500:-1])
    technology_min_indx = list(np.argsort(df[4])[:500])
    
    
    art_min_ind = [x for x in art_min_indx if (x in business_max_indx or 
                                               x in politics_max_indx or
                                               x in sports_max_indx or 
                                               x in technology_max_indx)]
    
    business_min_ind = [x for x in business_min_indx if (x in art_max_indx or
                                                         x in politics_max_indx or
                                                         x in sports_max_indx or
                                                         x in technology_max_indx)]
    
    politics_min_ind = [x for x in politics_min_indx if (x in business_max_indx or
                                                         x in art_max_indx or 
                                                         x in sports_max_indx 
                                                         or x in technology_max_indx)]
    
    sports_min_ind = [x for x in sports_min_indx if (x in business_max_indx 
                                                     or x in politics_max_indx 
                                                     or x in art_max_indx 
                                                     or x in technology_max_indx)]
    
    technology_min_ind = [x for x in technology_min_indx if (x in business_max_indx or
                                                             x in politics_max_indx or
                                                             x in sports_max_indx or
                                                             x in art_max_indx)]
    
    art_max_ind = [x for x in art_max_indx if (x not in business_max_indx and 
                                               x not in politics_max_indx and 
                                               x not in sports_max_indx and 
                                               x not in technology_max_indx)]
    
    business_max_ind = [x for x in business_max_indx if (x not in art_max_indx and
                                                         x not in politics_max_indx and
                                                         x not in sports_max_indx and
                                                         x not in technology_max_indx)]
    
    politics_max_ind = [x for x in politics_max_indx if (x not in business_max_indx and
                                                         x not in art_max_indx and
                                                         x not in sports_max_indx and
                                                         x not in technology_max_indx)]
    
    sports_max_ind = [x for x in sports_max_indx if (x not in business_max_indx and
                                                     x not in politics_max_indx and
                                                     x not in art_max_indx and
                                                     x not in technology_max_indx)]
    
    technology_max_ind = [x for x in technology_max_indx if (x not in business_max_indx and
                                                             x not in politics_max_indx and
                                                             x not in sports_max_indx and
                                                             x not in art_max_indx)]
    
    art_min = [tokens[x] for x in art_min_ind]
    business_min = [tokens[x] for x in business_min_ind]
    politics_min = [tokens[x] for x in politics_min_ind]
    sports_min = [tokens[x] for x in sports_min_ind]
    technology_min = [tokens[x] for x in technology_min_ind]
    
    art_max = [tokens[x] for x in art_max_ind]
    business_max = [tokens[x] for x in business_max_ind]
    politics_max = [tokens[x] for x in politics_max_ind]
    sports_max = [tokens[x] for x in sports_max_ind]
    technology_max = [tokens[x] for x in technology_max_ind]
    
    art_weight = [df[0][x] for x in art_max_ind]
    business_weight = [df[0][x] for x in business_max_ind]
    politics_weight = [df[0][x] for x in politics_max_ind]
    sports_weight = [df[0][x] for x in sports_max_ind]
    technology_weight = [df[0][x] for x in technology_max_ind]
    
    arts_tokenss = [x[0] for x in art]
    business_tokenss = [x[0] for x in busines]
    politics_tokenss = [x[0] for x in politic]
    sports_tokenss = [x[0] for x in sport]    
    technology_tokenss = [x[0] for x in technolog]
    

    art_max = [[x,arts_tokens[arts_tokenss.index(x)][2]] for x in art_max]
    business_max = [[x,business_tokens[business_tokenss.index(x)][2]] for x in business_max]
    politics_max = [[x,politics_tokens[politics_tokenss.index(x)][2]] for x in politics_max]
    sports_max = [[x,sports_tokens[sports_tokenss.index(x)][2]] for x in sports_max]
    technology_max = [[x,technology_tokens[technology_tokenss.index(x)][2]] for x in technology_max]
    
    arts_tokenss = [x[0] for x in art]
    business_tokenss = [x[0] for x in busines]
    politics_tokenss = [x[0] for x in politic]
    sports_tokenss = [x[0] for x in sport]    
    technology_tokenss = [x[0] for x in technolog]
    
    
    token_min = {"art_min":art_min,"business_min":business_min,"politics_min":politics_min,
                 "sports_min":sports_min,"technology_min":technology_min}
    token_max = {"art_max":art_max,"business_max":business_max,"politics_max":politics_max,
                 "sports_max":sports_max,"technology_max":technology_max}
    token_max_weight = {"art_max":art_weight,"business_max":business_weight,"politics_max":politics_weight,
                 "sports_max":sports_weight,"technology_max":technology_weight}
    
    train_classs = {"arts":token_max["art_max"],
                  "business":token_max["business_max"],
                  "politics":token_max["politics_max"],
                  "sports":token_max["sports_max"],
                  "technology":token_max["technology_max"]}
    
    train_classs_weight = {"arts":token_max_weight["art_max"],
                  "business":token_max_weight["business_max"],
                  "politics":token_max_weight["politics_max"],
                  "sports":token_max_weight["sports_max"],
                  "technology":token_max_weight["technology_max"]}
    
    return token_max,token_min,train_classs,train_classs_weight


def get_vocab():
    arts = np.load('arts.npz', allow_pickle=True)
    arts_tokens = arts['token_nonstop'].tolist()

    limit = [int(xx[1]) for xx in arts_tokens].index(20)
    arts_tokens = arts_tokens[:limit]
    del arts
    business = np.load('business.npz', allow_pickle=True)
    business_tokens = business['token_nonstop'].tolist()

    limit = [int(xx[1]) for xx in business_tokens].index(20)
    business_tokens = business_tokens[:limit]
    del business
    politics = np.load('politics.npz', allow_pickle=True)
    politics_tokens = politics['token_nonstop'].tolist()

    limit = [int(xx[1]) for xx in politics_tokens].index(20)
    politics_tokens = politics_tokens[:limit]
    del politics
    sports = np.load('sports.npz', allow_pickle=True)
    sports_tokens = sports['token_nonstop'].tolist()

    limit = [int(xx[1]) for xx in sports_tokens].index(20)
    sports_tokens = sports_tokens[:limit]
    del sports
    technology = np.load('technology.npz', allow_pickle=True)
    technology_tokens = technology['token_nonstop'].tolist()

    limit = [int(xx[1]) for xx in technology_tokens].index(20)
    technology_tokens = technology_tokens[:limit]
    del technology
    
    return business_tokens, arts_tokens, politics_tokens, sports_tokens, technology_tokens 
    
def get_test(hownet_dict):
    global test_classs,sememe_classs,test_clas,sem_class,line
    test = np.load('test.npz',allow_pickle=True)["clas"][0]
    arts = test["arts"]
    business = test["business"]
    politics = test["politics"]
    sports = test["sports"]
    technology = test["technology"]
    test_clas = [arts,business,politics,sports,technology]
    del test
    
    dict_pos_map = {'NOUN': "n",'VERB':"v",'ADJ' : "a",'ADV': "r"}
    sem_class = [[],[],[],[],[]]
    for indx,aclass in enumerate(test_clas):
        for index,line in enumerate(aclass):
            line = list(pos_tag(line,tagset="universal"))
            line = removal(line)
            sentence = [[x[0],"1",dict_pos_map[x[1]]] for x in line]
            sememe,weight = sememe_generation([sentence],hownet_dict,test_case=True)
            test_clas[indx][index] = [x[0] for x in line]
            sem_class[indx].append(sememe[0])
            
    test_classs = {"arts":test_clas[0],
                  "business":test_clas[1],
                  "politics":test_clas[2],
                  "sports":test_clas[3],
                  "technology":test_clas[4]}
    
    sememe_classs = {"arts":sem_class[0],
                  "business":sem_class[1],
                  "politics":sem_class[2],
                  "sports":sem_class[3],
                  "technology":sem_class[4]}
    return test_classs,sememe_classs
    
def pickle_it(files=[],names=None,instruction="wb"):
    if instruction == "wb":
        for index,file in enumerate(files):
            pickle_out = open(names[index] + ".pickle","wb")
            pickle.dump(file,pickle_out)
            pickle_out.close()
    else:
        for name in names:
            pickle_out = open(name + ".pickle", instruction)
            file = pickle.load(pickle_out)
            files.append(file)
            pickle_out.close()
        return files

def main():
    pass


if __name__ == "__main__":
    start_time = time.time()
    # train_class,test_class,sememes = pickle_it(names=["train_class","test_class","sememes"],instruction="rb")
    
    business_tokens, arts_tokens, politics_tokens, sports_tokens, technology_tokens = get_vocab()


    max_token, min_token, train_class, train_class_weight = generator(business_tokens, arts_tokens,
                                                   politics_tokens, sports_tokens, technology_tokens)
    
    business_tokens, arts_tokens, politics_tokens, sports_tokens, technology_tokens = train_class["business"],train_class["arts"],train_class["politics"],train_class["sports"],train_class["technology"]
    hownet_dict = OpenHowNet.HowNetDict()
    sememe,sememe_weight,unique_sememe,unique_sememe_weight = sememe_generation([business_tokens, arts_tokens, politics_tokens, sports_tokens, technology_tokens],hownet_dict)
    train_sememes={"arts":sememe[1],"business":sememe[0], "politics":sememe[2],
              "sports":sememe[3], "technology":sememe[4]}
    train_sememes_weights={"arts":sememe_weight[1],"business":sememe_weight[0], "politics":sememe_weight[2],
              "sports":sememe_weight[3], "technology":sememe_weight[4]}
    train_unique_sememes={"arts":unique_sememe[1],"business":unique_sememe[0], "politics":unique_sememe[2],
              "sports":unique_sememe[3], "technology":unique_sememe[4]}
    train_unique_sememes_weights={"arts":unique_sememe_weight[1],"business":unique_sememe_weight[0], "politics":unique_sememe_weight[2],
              "sports":unique_sememe_weight[3], "technology":unique_sememe_weight[4]}
    test_class,test_sememe = get_test(hownet_dict)
    for aclass in train_class:
        for idx,x in enumerate(train_class[aclass]): train_class[aclass][idx] = x[0]
    pickle_it([train_class,train_class_weight,test_class,test_sememe,train_sememes,train_sememes_weights,train_unique_sememes,train_unique_sememes_weights],["train_class","train_class_weight","test_class","test_sememe","train_sememes","train_sememes_weights","train_unique_sememes","train_unique_sememes_weights"])
    print(str(int(time.time()-start_time)) + " seconds has been spent.")
    