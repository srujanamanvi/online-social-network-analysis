"""
cluster.py
"""
import pickle
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def read_data():
    pkl_file = open('data.pkl', 'rb')
    tweets = pickle.load(pkl_file)
    pkl_file.close()
    return(tweets)

    ###TODO
    pass

def read_movie_names(filename):
    with open(filename) as f:
        names = [name.strip() for name in f]
    return(names)

    ###TODO
    pass


def create_graph(mtweets, movie_names):  
    graph = nx.Graph()  
    l =[]
    count_users = 0
    y = 0
    i = 1
    for tweets in mtweets:
        users = []
        for t in tweets:
            users.append(t['user']['screen_name'])           
            count_users+=len(users)
        for u in users:
            if u in graph: 
                count_users = count_users-1
                graph.add_edge(movie_names[y],u)          
            else:
                graph.add_node(u)
                graph.add_edge(movie_names[y],u)
        y+=1
    num_of_users = graph.number_of_nodes()-len(movie_names)
    f = open('users.txt','w',encoding='utf-8')
    f.write('The number of users collected: %d\n'%num_of_users)
    f.close()
    return(graph)

    ###TODO
    pass

def draw_network(graph, movie_name, filename):
    labels = {}    
    for node in graph.nodes():
        if node in movie_name:
            labels[node] = node
    position = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph,position,with_labels=False,node_color='c',node_size=40,width=0.5)
    nx.draw_networkx_labels(G=graph,pos=position,labels=labels,font_size=10,font_color='r')
    plt.savefig(filename)

    ###TODO
    pass

def get_communities(graph):
    H = graph.copy()
    i = 0
    def find_best_edge(G0):
        result = nx.edge_betweenness_centrality(G0)
        return sorted(result.items(), key=lambda x: x[1], reverse=True)[0][0]
    
    components = [c for c in nx.connected_component_subgraphs(H)]
    while len(components) < 4:
        edge_to_remove = find_best_edge(H)
        H.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(H)]
        i = i+1
    n = 0
    num_of_communities = len(components)
    for x in range(len(components)):    
        n += len(components[x])
    avg_users_per_community = n/len(components)
    f = open('community.txt','w',encoding='utf-8')
    f.write('The number of communities discovered:%d\n'%num_of_communities)
    f.write('Average number of users per community:%d\n'%avg_users_per_community)
    f.close()
    return(components)
        
    ###TODO
    pass

def main():
    mtweets = read_data()
    num_of_messages = 0
    for i in mtweets:
        num_of_messages +=len(i)
    f = open('messages.txt','w',encoding='utf-8')
    f.write('The number of messages collected:%d\n'%num_of_messages)
    f.close()
    movie_names = read_movie_names('movies.txt')
    graph = create_graph(mtweets, movie_names)
    print('The number of edges in the graph are:%s'% graph.number_of_edges())
    print('The number of nodes in the graph are:%s'% graph.number_of_nodes())
    draw_network(graph, movie_names, 'network.png')
    print('graph drawn')
    components = get_communities(graph)
    for i,c in enumerate(components):
        draw_network(c, movie_names, str(i)+'.png')
       

if __name__ == '__main__':
    main()