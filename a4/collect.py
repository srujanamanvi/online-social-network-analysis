"""
collect.py
"""
import requests
from TwitterAPI import TwitterAPI
import pickle
import time
import sys

consumer_key = 'PxOyfDGW6LiaKRDMEDQRs1CKW'
consumer_secret = 'gzQa7GzYQe4RObD7by7W8s9CEDfZ6hEvVDh1ya8NwCXN3qq47T'
access_token = '768560420231651328-Q0TjMUzKZygOqSErVWdKGtycYanfEgC'
access_token_secret = 'sAu48LKwFKNtSC9YWPU4BxFhQfsZH60aUFHe3MNhDACp0'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


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
            time.sleep(61 * 5)
    ###TODO
    pass
           
def read_movie_names(filename):
    with open(filename) as f:
        names = [name.strip() for name in f]
    return(names)

    ###TODO
    pass

def get_movie_tweets(twitter, movie_names):
    l = []
    for movie in movie_names:
        response = robust_request(twitter,"search/tweets",{'q': movie,'language':'en','count':100}, max_tries = 5)
        res = [r for r in response] 
        while len(res) < 130:
            max_id = res[len(res)-1]['id']
            response = robust_request(twitter,"search/tweets",{'q': movie,'language':'en','max_id':max_id,'count':50}, max_tries = 5)
            for r in response:
                res.append(r)
        l.append(res)
    return(l)

    ###TODO
    pass


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    print('Established Twitter connection.')
    movie_names = read_movie_names('movies.txt')
    print('collecting data...takes about 10 secs')
    movie_tweets = get_movie_tweets(twitter, movie_names)
    output = open('data.pkl', 'wb')
    pickle.dump(movie_tweets, output)
    output.close()
    print('data saved')       

if __name__ == '__main__':
    main()

