import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('processed_input.txt',delimiter="\t", encoding = "utf-8")
index = [i for i in range(0,len(df['Word']))]
df = df.fillna(method='ffill')


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p) for w, p in zip(s['Word'].values.tolist(), 
                                                           s['POS'].values.tolist()) 
                                                          ]
        self.grouped = self.data.groupby('Sentence#').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None
getter = SentenceGetter(df)

sentences = getter.sentences

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,  
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({

            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, label in sent]
def sent2tokens(sent):
    return [token for token, postag, label in sent]

y_pred=[]
X = [sent2features(s) for s in sentences]
print(X)


with open("ner.pkl","rb") as f:
    crf2 = pickle.load(f)


#y_pred = crf.predict(X)
y_pred = crf2.predict(X)

print(y_pred)
y=[]

for i in y_pred:
    for j in i:
        y.append(j)

df['Tag'] = y

#f = open('output.txt' , mode= 'w', encoding='utf-8')
df.to_csv('output.txt', sep='\t', encoding='utf-8')


