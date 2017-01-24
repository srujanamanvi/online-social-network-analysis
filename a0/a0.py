# coding: utf-8

"""
CS579: Assignment 0
Collecting a political social network

In this assignment, I've given you a list of Twitter accounts of 4
U.S. presedential candidates.

The goal is to use the Twitter API to construct a social network of these
accounts. We will then use the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

1. Create an account on [twitter.com](http://twitter.com).
2. Generate authentication tokens by following the instructions [here](https://dev.twitter.com/docs/auth/tokens-devtwittercom).
3. Add your tokens to the key/token variables below. (API Key == Consumer Key)
4. Be sure you've installed the Python modules
[networkx](http://networkx.github.io/) and
[TwitterAPI](https://github.com/geduldig/TwitterAPI). Assuming you've already
installed [pip](http://pip.readthedocs.org/en/latest/installing.html), you can
do this with `pip install networkx TwitterAPI`.

OK, now you're ready to start collecting some data!

I've provided a partial implementation below. Your job is to complete the
code where indicated.  You need to modify the 10 methods indicated by
#TODO.

Your output should match the sample provided in Log.txt.
"""

# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import collections
import sys
import time
import requests
import json
import urllib.request
import numpy as np
from TwitterAPI import TwitterAPI
from itertools import combinations
from operator import itemgetter
import matplotlib.pyplot as plt

consumer_key = 'PxOyfDGW6LiaKRDMEDQRs1CKW'
consumer_secret = 'gzQa7GzYQe4RObD7by7W8s9CEDfZ6hEvVDh1ya8NwCXN3qq47T'
access_token = '768560420231651328-Q0TjMUzKZygOqSErVWdKGtycYanfEgC'
access_token_secret = 'sAu48LKwFKNtSC9YWPU4BxFhQfsZH60aUFHe3MNhDACp0'


# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    with open('candidates.txt') as f:
        names = [name.strip() for name in f]
    return(names)
    ###TODO
    pass


# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi','twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    response = robust_request(twitter,"users/lookup",{'screen_name':screen_names},max_tries = 5)
    return(response)
    ###TODO
    pass


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids

    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    response = robust_request(twitter,"friends/ids",{'screen_name':screen_name,'count':5000}, max_tries=5)
    return(sorted(response))
    ###TODO
    pass


def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    friends = []
    u = [d['screen_name'] for d in users]
    for x in u:
        f = get_friends(twitter,x)
        friends.append(f)
    for d,num in zip(users,friends):
        d['friends'] = num
    [dict(d, friends=n) for d, n in zip(users, friends)]
    
    ###TODO
    pass


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    l= []
    keys = [r['screen_name'] for r in users]
    friend_c = [r['friends_count'] for r in users]
    dictionary = dict(zip(keys, friend_c))
    dic_new = sorted(dictionary.items(),key=itemgetter(1),reverse=True)
    for x in dic_new:
        print(x[0],x[1])
    
    ###TODO
    pass             
   

def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    l = [tok['friends'] for tok in users]
    lt = [val for sublist in l for val in sublist]
    c = Counter(lt)
    return c
    ###TODO
    pass


def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.

    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    my_list= [d['screen_name'] for d in users]
    friend_list = [d['friends'] for d in users]
    l = []
    x = []
    f = []
    combo= list(combinations(my_list,2))
    for q in combo:
        a =list(q)
        l.append(a)
    for pair in combinations(friend_list,2):
        z = len(set(pair[0])&set(pair[1]))
        x.append(z)        
    for y,num in zip(l,x):
        y.append(num)
    for x in l:
        b=tuple(x)
        f.append(b)
    f.sort(key=lambda x: x[1])
    f.sort(key=lambda x: x[0])
    return(sorted(f,key=itemgetter(2), reverse=True))
    ###TODO
    pass


def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_name of the one Twitter user followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A string containing the single Twitter screen_name of the user
        that is followed by both Hillary Clinton and Donald Trump.
    """
    new_d = {}
    l = []
    names = [d['screen_name'] for d in users]
    friend_list = [d['friends'] for d in users]
    for x,f in zip(names,friend_list):
        new_d[x] = f
    l.append(new_d['HillaryClinton'])
    l.append(new_d['realDonaldTrump'])
    result_user = set(l[0])&set(l[1])
    rc = list(result_user)
    response = robust_request(twitter,"users/show",{'id':rc},max_tries=5)
    return([u['screen_name'] for u in response])
    ###TODO
    pass


def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)


        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    graph = nx.Graph()
    u = [a['screen_name'] for a in users]
    L = friend_counts.items()
    new_list = [x[0] for x in L if x[1]>1]
    for n in u:
        graph.add_node(n)
    y = 0
    friends_list = [d['friends'] for d in users]
    for index in range(len(friends_list)):
        for f in friends_list[index]:
            if f in new_list:
                graph.add_node(f)
                graph.add_edge(u[y],f)
        y+=1
    return(graph) 
    ###TODO
    pass


def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    candidates = [a['screen_name'] for a in users]
    labels = {}    
    for node in graph.nodes():
        if node in candidates:
            labels[node] = node
    position = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph,position,with_labels=False,node_color='c',node_size=40,width=0.5)
    nx.draw_networkx_labels(G=graph,pos=position,labels=labels,font_size=10,font_color='r')
    plt.savefig(filename)
    ###TODO
    pass


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))
    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()

# That's it for now! This should give you an introduction to some of the data we'll study in this course.