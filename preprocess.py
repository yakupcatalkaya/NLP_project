# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:59:51 2022

@author: yakupcatalkaya
"""

import numpy as np
import os
import string
import re
import time
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
import nltk
import h5py

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
start_time = time.time()
cur_direc = os.getcwd()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = list(set(stopwords.words("english")))
translator = str.maketrans('', '', string.punctuation)
stop_words = [stop_word.translate(translator) for stop_word in stop_words]
file_holder = []
file_names = []
listfile = os.listdir(cur_direc)[::-1]

dict_pos_map = {
    'NOUN': "n",
    'VERB':"v",
    'ADJ' : "a",
    'ADV':"r"  }

def removal(words):
    new_words = []
    for word in words:
        if word[1] in list(dict_pos_map.keys()):
            new_words.append(word)
    return new_words

def replacer(input_text):
    input_text = input_text.replace("\n"," ").replace("â",'"')
    input_text = " ".join(input_text.split())
    input_text = re.sub(r'\d+', '', input_text)
    translator = str.maketrans('', '', string.punctuation)
    input_text = input_text.translate(translator)
    for spec_item in list(['"',".",",",";","?",";","!",":","'","(",")",
                           "[","]","...","-","/","@","{","}","*","_","’",
                           "”","“","‘","—","+","&","$","=","|","×","£",
                           "`","′","ϰ","η","τ","β","μ","σ","ν","κ","ε",
                           "ι","ς","ο","π","ו","ח","~","#","%"]):
        input_text = input_text.replace(spec_item,"")
    input_text = input_text.replace("æ","a").replace("é","e").replace("à","a")
    input_text = input_text.replace("â","a").replace("œ","a").replace("ë","e")
    input_text = input_text.replace("α","a").replace("è","e").replace("ö","o")
    input_text = input_text.encode("ascii", "ignore")
    input_text = input_text.decode().lower()
    return input_text

def preprocess():
    global file,set_wout_stopword,set_token
    for file_dir in listfile:
        file_dir = cur_direc + "/" + file_dir
        if ".txt" in file_dir:
            print(file_dir)
            file = open(file_dir, "r",encoding=("utf-8")).read()
            new_file = replacer(file)
            file_holder.append(new_file)
            file_names.append(file_dir)
            del new_file
            del file
    for file_index,file in enumerate(file_holder):
        print(file_index,file_names[file_index])
        
        set_token = []
        set_wout_stopword = []
        file = word_tokenize(file)
        file = pos_tag(file,tagset="universal")
        file = removal(file)
        file = [[lemmatizer.lemmatize(word[0],pos=dict_pos_map[word[1]]),dict_pos_map[word[1]]] for word in file]
        file.sort(key=lambda x:x[0],reverse=True)
        file = np.array(file,dtype='object')
        freq=0
        old_token, old_tag = file[0]
        for token_index,tokenn in enumerate(file):
            token, tag = tokenn
            if old_token == token and old_tag == tag:
                freq += 1
            else:
                if not str(old_token) in stop_words:
                    set_wout_stopword.append([old_token,freq,old_tag])
                set_token.append([old_token,freq,old_tag])
                freq = 1
                old_token,old_tag = token,tag
        set_token.sort(key=lambda x: x[1],reverse=True)
        set_wout_stopword.sort(key=lambda x: x[1],reverse=True)
        set_token = np.array(set_token,dtype='object')
        set_wout_stopword = np.array(set_wout_stopword,dtype='object')
        
        np.savez_compressed(file_names[file_index].split(".")[0],token=set_token,
                            token_nonstop=set_wout_stopword,file_string=file)
        
        del set_token
        del set_wout_stopword
        del file
        
preprocess()

print("Time spent: ",int(time.time()-start_time)," seconds.")
        