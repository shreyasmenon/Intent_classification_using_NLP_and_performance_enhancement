# -*- coding: utf-8 -*-
"""

@author: Shreyas Menon
"""

import os
import nltk
import pandas as pd
import numpy as np
import string
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim.models.doc2vec import TaggedDocument
from random import shuffle
from gensim.models import doc2vec
from nltk.collocations import *

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from operator import itemgetter
import re
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from gensim import corpora
import pyLDAvis.gensim
from scipy.spatial import distance
from sklearn.metrics.pairwise import linear_kernel
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tokenize import sent_tokenize
import random
from enum import Enum

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import re
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
import spacy
from nltk.tokenize import TweetTokenizer

if 'spacy_nlp' not in globals():
    spacy_nlp = spacy.load('en_core_web_lg') 



phone_pattern = '(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})' 
    
#address_pattern = r'(\d[0-9]{1,4} [\w]{1,10} (st|street|road|rd|blvd|boulevard|ave|avenue) [\w+\s\w+]{0,} (nj|ny|ct|new jersey|new york|connecticut) (\d{5})*)'
address_pattern = r'([0-9]{1,4} [\w]{1,20}(.*)(st|street|road|rd|blvd|boulevard|ave|avenue|ct|drive|dr|lane|ace|way)(.{0,20})(nj|ny|ct|new jersey|new york|connecticut)[\s]{0,}[\d]{0,5})'

account_number_pattern = "(\d{5}[-\.\s]??\d{6}[-\.\s]??\d{2}[-\.\s]??\d{1}|\(\d{5}\)\s*\d{6}[-\.\s]??\d{2}[-\.\s]??\d{1})"

def extract_metainfo(response:str):
    
    meta_response = response.lower()
    
    image_link_pattern1 = '(https://t.co/[a-zA-Z0-9]{10})'
    
    image_link_pattern2 = '(t.co/[a-zA-Z0-9]{10})'
    
    #extract account numbers
    account_numbers = re.findall(account_number_pattern,meta_response)
    #replace all account numbers with empty characters are they should not be confused with phone numbers
    meta_response = re.sub(account_number_pattern,'accountnumberprovided',meta_response)
    
    meta_response = re.sub(image_link_pattern1,' ',meta_response)
    meta_response = re.sub(image_link_pattern2,' ',meta_response)
    
    phone_numbers = re.findall(phone_pattern,meta_response)
    addresses = re.findall(address_pattern,meta_response)
    
    if len(phone_numbers) > 0:
        phone_number = phone_numbers[0][0]
        meta_response = re.sub(phone_pattern,' phoneprovided ',meta_response)
        
    if len(addresses) > 0:
        addr = addresses[0][0]
        meta_response = meta_response.replace(addr,' addressprovided ')
        #Added layer of assurance
        meta_response = re.sub(address_pattern,' addressprovided ',meta_response)
        

    
    meta_response = meta_response.strip()
    
    #stripping handles
    tknzr = TweetTokenizer(strip_handles=True)    
    tweet_tokens = tknzr.tokenize(meta_response)
    tweet_tokenized = " ".join(tweet_tokens) 
        
    #Get other entities
    spacy_doc = spacy_nlp(tweet_tokenized)
    
    #extracting person
    list_human_names = [ entity.text for entity in spacy_doc.ents if entity.label_ == 'PERSON' and str(entity.text).isalpha() ]
    
    if len(list_human_names) > 0 :     
        for name in list_human_names:
            meta_response = meta_response.replace(name,' nameprovided ')
                     
    return meta_response


def get_wordnet_pos(pos_tag):
    
    # if pos tag starts with 'J'
    if pos_tag.startswith('J'):
        # return wordnet tag "ADJ"
        return wordnet.ADJ
    
    # if pos tag starts with 'V'
    elif pos_tag.startswith('V'):
        # return wordnet tag "VERB"
        return wordnet.VERB
    
    # if pos tag starts with 'N'
    elif pos_tag.startswith('N'):
        # return wordnet tag "NOUN"
        return wordnet.NOUN
    
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        # be default, return wordnet tag "NOUN"
        return wordnet.NOUN



def tokenize_lemmatize(df_queries, use_stopwords = True , extract_entities = True):
    
    list_preprocessed = []
    stop_words = stopwords.words('english')
    
    #Add relevant words
    stop_words.append('https')
    stop_words.append('http')
    stop_words.append('thank')
    stop_words.append('thanks')
    stop_words.append('hi')
    stop_words.append('dm')
    stop_words.append('ok')
    stop_words.append('okay')
    stop_words.append('twigmg')
    stop_words.append('pbs')
    stop_words.append('jpg')
    stop_words.append('com')
    
    
    #Remove words from list
    stop_words.remove('not')
    stop_words.remove('nor')
    stop_words.remove('no')
    stop_words.remove('ain')
    stop_words.remove("aren't")
    stop_words.remove("couldn't")    
    #stop_words.remove("couldn'")             
    #stop_words.remove("didn'")
    #stop_words.remove("doesn'")
    stop_words.remove("doesn't")
    #stop_words.remove("don'")
    stop_words.remove("don't")
    stop_words.remove("hadn")
    stop_words.remove("hadn't")
    stop_words.remove("hasn")
    stop_words.remove("hasn't")
    stop_words.remove("haven")
    stop_words.remove("haven't")
    #stop_words.remove("isn'")
    stop_words.remove("isn't")    
    stop_words.remove("mightn")
    stop_words.remove("mightn't")
    stop_words.remove("mustn")
    stop_words.remove("mustn't")
    stop_words.remove("needn't")
    stop_words.remove("off")
    stop_words.remove("shan't")
    #stop_words.remove("shan'")
    stop_words.remove("should've")
    #stop_words.remove("shouldn'")
    stop_words.remove("shouldn't")
    #stop_words.remove("wasn''")    
    stop_words.remove("wasn't")
    #stop_words.remove("weren'")   
    stop_words.remove("weren't")    
    stop_words.remove("will")
    #stop_words.remove("wouldn'")
    #stop_words.remove("won'")
    stop_words.remove("won't")
    #stop_words.remove("wouldn'")
    stop_words.remove("wouldn't")
    stop_words.remove("down")
    
    printable = set(string.printable)

    
    for response in df_queries:
        #sample_response = response
        #keep only unicode characters    
        response = ''.join(filter(lambda x: x in printable, response))
        
        if extract_entities:
            response = extract_metainfo(response)
         
        tokeinzed_response = nltk.word_tokenize(response)
        if use_stopwords:
            tokeinzed_response_lower = [token.lower() for token in tokeinzed_response if len(token) > 1 and token not in stop_words]
        else:
            tokeinzed_response_lower = [token.lower() for token in tokeinzed_response if len(token) > 1]
        tokenized_without_punctuations =[token.strip(string.punctuation) for token in tokeinzed_response_lower]
        tokenized_without_punctuations_and_spaces = [token.strip() for token in tokenized_without_punctuations if token.strip()!='']
        tagged_tokens= nltk.pos_tag(tokenized_without_punctuations_and_spaces)
        
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_words = [wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for (word, tag) in tagged_tokens if word not in string.punctuation]       
        lemmatized_string = " ".join(lemmatized_words)
        
        
        if len(lemmatized_string) > 0:
            list_preprocessed.append(lemmatized_string)
        else:
            list_preprocessed.append('Empty String')
            
    return list_preprocessed
            
            
            
def extract_metainfo(response:str):
    
    meta_response = response.lower()
    
    image_link_pattern1 = '(https://t.co/[a-zA-Z0-9]{10})'
    
    image_link_pattern2 = '(t.co/[a-zA-Z0-9]{10})'
    
    #extract account numbers
    account_numbers = re.findall(account_number_pattern,meta_response)
    #replace all account numbers with empty characters are they should not be confused with phone numbers
    meta_response = re.sub(account_number_pattern,'accountnumberprovided',meta_response)
    
    meta_response = re.sub(image_link_pattern1,' ',meta_response)
    meta_response = re.sub(image_link_pattern2,' ',meta_response)
    
    phone_numbers = re.findall(phone_pattern,meta_response)
    addresses = re.findall(address_pattern,meta_response)
    
    if len(phone_numbers) > 0:
        phone_number = phone_numbers[0][0]
        meta_response = re.sub(phone_pattern,' phoneprovided ',meta_response)
        
    if len(addresses) > 0:
        addr = addresses[0][0]
        meta_response = meta_response.replace(addr,' addressprovided ')
        #Added layer of assurance
        meta_response = re.sub(address_pattern,' addressprovided ',meta_response)
        

    
    meta_response = meta_response.strip()
    
    #stripping handles
    tknzr = TweetTokenizer(strip_handles=True)    
    tweet_tokens = tknzr.tokenize(meta_response)
    tweet_tokenized = " ".join(tweet_tokens) 
        
    #Get other entities
    spacy_doc = spacy_nlp(tweet_tokenized)
    
    #extracting person
    list_human_names = [ entity.text for entity in spacy_doc.ents if entity.label_ == 'PERSON' and str(entity.text).isalpha() ]
    
    if len(list_human_names) > 0 :     
        for name in list_human_names:
            meta_response = meta_response.replace(name,' nameprovided ')
                     
    return meta_response

def tf_idf(list_queries , min_df = 5 ,  stopwords = None ):
    
    tfidf_input = list_queries
                 
    if stopwords == None:
        tfidf_vect = TfidfVectorizer(min_df = min_df )
    else:
        tfidf_vect = TfidfVectorizer(min_df = min_df , stop_words = 'english' ) 
    
    
    dtm = tfidf_vect.fit_transform(tfidf_input)
    
    #print(tfidf_vect.get_feature_names())
    
    return dtm
