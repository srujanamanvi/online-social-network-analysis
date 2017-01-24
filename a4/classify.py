"""
classify.py
"""
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle

def read_data():  
    pkl_file = open('data.pkl', 'rb')
    tweets = pickle.load(pkl_file)
    pkl_file.close()
    return(tweets)

    ###TODO
    pass

def get_csv(tweets):    
    with open('dict.csv', 'w',newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for t in tweets:
            for j in t:
                a = []
                a.append(0)
                a.append(j['text'])
                writer.writerow(a)
    return(writer)
       
    ###TODO
    pass

def load_file():
    with open('dict.csv',encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        data =[]
        polarity = []
        for row in reader:
            if row[0] and row[1]:
                polarity.append(row[0])
                data.append(row[1])
        docs = np.array(data)
        labels = np.array(polarity)
    return(docs,labels)
    
    ###TODO
    pass
        
def preprocess(data):
    count_vectorizer = CountVectorizer(binary='true')
    data = count_vectorizer.fit_transform(data)
    vocab = count_vectorizer.vocabulary_
    tfidf_data = TfidfTransformer(use_idf=False).fit_transform(data)
    return(tfidf_data, vocab)

    ###TODO
    pass

def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)
    
    ###TODO
    pass

def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return(np.mean(accuracies))
    
    ###TODO
    pass

def train_test_split(ratings):
    test = set(range(len(ratings))[::10])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return train, test

def main():
    #tweets = read_data()
    #data = get_csv(tweets)
    #print('data copied to csv')
    d_train = []
    d_test = []
    d_label = []
    d_test_label = []
    data,polarity = load_file()
    mat, vocab = preprocess(data)
    train, test = train_test_split(data)
    for i in train:
        d_train.append(data[i])
        d_label.append(polarity[i])
    for j in test:
        d_test.append(data[j])
        d_test_label.append(polarity[j])
    train_mat, voc = preprocess(d_train)
    count_vectorizer = CountVectorizer(vocabulary=voc)
    test_mat = count_vectorizer.fit_transform(d_test)
    clf = LogisticRegression()
    clf.fit(train_mat, d_label) 
    pred = clf.predict(test_mat)
    negative = []
    positive = []
    neutral = []
    for i,p in enumerate(pred):
        if p =='0':
            negative.append(d_test[i])
        elif p == '2':
            neutral.append(d_test[i])
        elif p == '4':
            positive.append(d_test[i])
    f = open('class.txt','w')
    unique, counts = np.unique(pred, return_counts=True)
    for x in range(len(unique)):
        f.write('The number of instances in class %s is: %s\n'%(unique[x],counts[x])) 
    f.close()
    f = open('example.txt','w',encoding='utf-8')
    f.write('The example for negative class is: %s\n'%negative[0])
    f.write('The example for positive class is: %s\n'%positive[0])
    f.write('The example for neutral class is: %s\n'%neutral[0])
    f.close()
    avg_accuracy = cross_validation_accuracy(clf, mat, polarity, 5)
    print('testing accuracy=%f' % avg_accuracy) 
    
if __name__ == '__main__':
    main()