# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    movies['tokens'] = ""
    gen = movies.genres
    tok = [tokenize_string(d) for d in gen]
    for i,t in enumerate(tok):
        movies.set_value(i, 'tokens', t)
    return(movies)
    
    ###TODO
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    movies['features'] = ""
    list_of_tf = []
    vocab = {}
    doc = 0
    row = []
    col = []
    data = []
    max_val = {}
    x = movies['tokens'].tolist()
    for i in x:
        token_freq = Counter(i)
        l = [(k,v) for k,v in token_freq.items()]
        max_k = max([m[1] for m in l])
        max_val[doc] = max_k
        doc+=1
        list_of_tf.append(l) 
    N = len(list_of_tf)
    df = Counter(f[0] for sublist in list_of_tf for f in sublist)
    for i,order in enumerate(sorted(df)):
        vocab[order] = i
    val = 0
    y= 0
    
    tf_list = []
    for i,feat in enumerate(list_of_tf):
        feat_l = []
        for tup in feat:
            y = N/df[tup[0]]
            tf_idf = tup[1]/max_val[i]*math.log10(y)
            feat_l.append((tup[0],tf_idf))  
        tf_list.append(feat_l)
        
    for i,feat in enumerate(tf_list):
        for tup in feat:
            row.append(i)
            data.append(tup[1])
            col.append(vocab[tup[0]])
    X = csr_matrix((data, (row, col)))
    for i in range(len(set(row))):
        movies.set_value(i, 'features', csr_matrix(X[i]))
    return((movies,vocab))
    
    ###TODO
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    cos_sim = a.T.dot(b).sum()/math.sqrt(sum(a.data ** 2)) * math.sqrt(sum(b.data ** 2))
    return(cos_sim)

    ###TODO
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """

    l = []
    for i,row in ratings_test.iterrows():
        cs = []
        csp = []
        u = row['userId']
        m = movies.loc[movies['movieId']==row['movieId'], 'features'].iloc[0] 
        for j,r in ratings_train[ratings_train.userId==u].iterrows(): 
            mov = movies.loc[movies['movieId']==r['movieId'], 'features'].iloc[0]
            cos_sim = cosine_sim(mov, m)
            if cos_sim>=0:
                cs.append(cos_sim)
                csp.append(cos_sim*r['rating'])  
        if not cs:
            weighted_avg = ratings_train.rating[ratings_train.userId==u].mean()
        else:    
            weighted_avg = sum(csp)/sum(cs)
        l.append(weighted_avg)     
    x = np.array(l)    
    return(x)
 
    ###TODO
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))  
    predictions = make_predictions(movies, ratings_train, ratings_test)  
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])

if __name__ == '__main__':
    main()
