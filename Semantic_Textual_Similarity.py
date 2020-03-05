#Importing library for importig dataset

import numpy as np
import pandas as pd


#Importing the dataset

dataset=pd.read_csv('C:/Users/Arushi Saxena/OneDrive/Desktop/semantic textual similarity/Text_Similarity_Dataset.csv')
X=dataset.iloc[:,1:2].values

#Importing the libraries for word2vec model

import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#Load the google's pretrained word2vec model
word2vec_model=gensim.models.KeyedVectors.load_word2vec_format('C:/Users/Arushi Saxena/OneDrive/Desktop/semantic textual similarity/GoogleNews-vectors-negative300.bin.gz', binary=True)

#List of words

index2word_set=set(word2vec_model.wv.index2word)

#Function to fiind out average vector of the paragraph
def avg_sen_vector(words, model, num_feature, index2word_set):
    
    featureVec= np.zeros((num_feature,), dtype="float32")
    nwords = 0
    
    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
            
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    
    return featureVec


from scipy import spatial
list=[]
for i in range(0,4023):
    
    #get average vector of text 1
    Paragraph_1 =dataset['text1'][i]
    Paragraph_1_avg_vector = avg_sen_vector(Paragraph_1.split(), model=word2vec_model, num_feature=300,index2word_set=index2word_set)
    Paragraph_1_avg_vector=Paragraph_1_avg_vector.reshape(-1,1)

    #get average vector of text2
    Paragraph_2 =dataset['text2'][i]
    Paragraph_2_avg_vector = avg_sen_vector(Paragraph_2.split(), model=word2vec_model, num_feature=300,index2word_set=index2word_set)
    Paragraph_2_avg_vector=Paragraph_2_avg_vector.reshape(-1,1)
    
    #get cosine similarity between text1 and text2
    para1_para2_similarity = 1-spatial.distance.cosine(Paragraph_1_avg_vector,Paragraph_2_avg_vector)
    list.append(para1_para2_similarity)

#Creating Csv file for the similarity score    
output = pd.DataFrame((list),columns=['Similarity_Score'])
output.to_csv("C:/Users/Arushi Saxena/OneDrive/Desktop/semantic textual similarity/Textual_similarity_score.csv")
    