'''
Network analysis and visualization
'''

#import libraries
from tqdm import tqdm
import pandas as pd
import networkx as nx
from ast import literal_eval
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#define functions
def format_df(embassy_df):

    '''
    apply literal_eval to selected columns of the dataframe
    add channel_username, channel_id and channel_title column for ease of processing
    convert the 'date' column to datetime, set as index, sort to allow slicing
    '''

    embassy_df['_chat'] = embassy_df['_chat'].apply(literal_eval)
    embassy_df['channel_id'] = embassy_df['_chat'].apply(lambda x: x['id'])
    embassy_df['channel_title'] = embassy_df['_chat'].apply(lambda x: x['title'])
    embassy_df['channel_username'] = embassy_df['_chat'].apply(lambda x: x['username'])

    embassy_df['date'] = pd.to_datetime(embassy_df['date']) 
    embassy_df = embassy_df.set_index('date')
    embassy_df = embassy_df.sort_index() 

    return embassy_df


def construct_digraph(sources, targets):
    '''
    Construct a directed graph from list of source nodes and target nodes
    '''
    G = nx.DiGraph()
    for source, target in zip(sources, targets):
        if G.has_edge(source, target):
            G[source][target]['weight'] += 1
        else:
            G.add_edge(source, target, weight=1)

    return G

def plot_sankey(H, output_path):
    '''
    Start from a directed acyclic graph
    Sort nodes according to topological sort
    Create a Sankey diagram of the information flow within the network
    Produces a PNG images with a Sankey diagram
    '''

    #determine order of nodes using topological sorting
    topological_order = list(nx.topological_sort(H))

    #get data
    edges = list(H.edges(data=True))
    source_nodes = [u for u, v, d in edges]
    target_nodes = [v for u, v, d in edges]
    values = [d['weight'] for u, v, d in edges]

    #prepare the data for plotly
    node_labels = topological_order
    source_indices = [node_labels.index(u) for u in source_nodes]
    target_indices = [node_labels.index(v) for v in target_nodes]

    #create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values
        )
    )])

    #save the sankey diagram 
    fig.update_layout(title_text="Information flow", font_size=10, width = 1200, height =  10 + len(node_labels) * 11)
    fig.write_image(output_path, scale = 3) #change scale to increase resolution 


def filter_graph(G, min_edge_weight):
    '''
    remove self-loops
    filter graph by minimum edge weight
    '''

    G.remove_edges_from(nx.selfloop_edges(G))

    #filter the graph by minimum edge weight
    n = min_edge_weight
    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data['weight'] > n:
            H.add_edge(u, v, weight=data['weight'])

    return H

def get_graph_metrics(G):
    ''' 
    get metrics for a given graph G
    returns number of nodes, number of edges, number of cycles, list of cycles, DAG or not (True or False)
    '''
    is_DAG = nx.is_directed_acyclic_graph(G) #check if the graph is a DAG
    edge_count = len(G.edges) #get the number of nodes
    node_count = len(G.nodes) #get the number of edges 
    
    print('identify biderectional edges')
    bidirectional_edges = []
    for u, v in tqdm(G.edges):
        if G.has_edge(v, u): #check if reverse edge exists
            bidirectional_edges.append((u, v))
    bidirectional_edge_count = len(bidirectional_edges)

    return {'node_count':node_count, 'edge_count':edge_count, 'is_DAG': is_DAG, 'bidirectional_edge_count':bidirectional_edge_count}


def plot_message_forwarding_network(G, output_path):
    '''
    create plot of message forwarding network
    '''

    #visualize the graph
    plt.figure(figsize=(12, 12))

    #calculate the out-degree for each node, get max out degree for scaling
    out_degrees = dict(G.out_degree())
    node_sizes = [out_degrees[node] * 9 for node in G.nodes()]

    #get the nodes with the highest out degree 
    top_degree_nodes = sorted(out_degrees, key=out_degrees.get, reverse=True)[:4]
    print('top degree nodes', top_degree_nodes)

    #set edge colors
    edge_color = [(0.9, 0.9, 0.9, 0.1) for _ in G.edges()]

    #set node colours
    node_colours = []
    for node in G.nodes:
        if node in embassy_channels:
            node_colours.append('blue')
        else:
            node_colours.append('red')

    #node positioning, draw network
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed = 10) #spring layout with fixed seed 
    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color=node_colours, alpha=0.6, edge_color = edge_color, arrowsize=2)

    #add labels for high degree nodes 
    degree_labels = {node: node for node in top_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels=degree_labels, font_size=10, font_color='black')

    #add labels for embassy nodes 
    embassy_labels = {node: node for node in [channel for channel in embassy_channels if channel != 'rusembrwanda']} #skip channel 'rusembr
    nx.draw_networkx_labels(G, pos, labels=embassy_labels, font_size=6, font_color='black')

    #add legend to the plot
    embassy_patch = mpatches.Patch(color='blue', label='embassy channels')
    non_embassy_patch = mpatches.Patch(color='red', label='non-embassy channels')
    plt.legend(handles=[embassy_patch, non_embassy_patch], loc='lower right')

    #add title to the plot
    plt.title('Russian embassies on Telegram \n Message-forwarding network (2020-2024)')

    #tight layout
    plt.tight_layout()

    #save the figure
    plt.savefig(output_path, bbox_inches = 'tight')

    #show figure
    plt.show()


def get_graph_metrics(G):
    ''' 
    get metrics for a given graph G
    returns number of nodes, number of edges, number of cycles, list of cycles, DAG or not (True or False)
    '''

    #check if network is DAG
    is_DAG = nx.is_directed_acyclic_graph(G)  # check if the graph is a DAG
    edge_count = len(G.edges)  # get the number of edges
    node_count = len(G.nodes)  # get the number of nodes 
    
    #find cycles using SCCs if the graph is not acyclic
    cycles = []
    cycle_count = 0
    if not is_DAG:

        #find strongly connected components
        sccs = list(nx.strongly_connected_components(G))  
        
        #each SCC with more than 1 node has at least one cycle
        for scc in sccs:
            if len(scc) > 1:  #if SCC has more than 1 node, it contains cycles
                cycles.append(list(scc))
        cycle_count = len(cycles)

    return {
        'node_count': node_count,
        'edge_count': edge_count,
        'is_DAG': is_DAG,
        'SCC_count': cycle_count,
        'SCC': cycles
    }

#get list of network components and network metrics for different min edge weights
def store_graph_metrics(G, output_path):
    ''' 
    filter graph for edge weights in range (0,21)
    store metrics about SCCs at different edge weights
    store as csv
    '''

    graph_metrics_dicts = []
    for i in tqdm(range(0, 21)):
        I = filter_graph(G,i)
        graph_metrics = get_graph_metrics(I)
        graph_metrics['min_edge_weight'] = i + 1 
        graph_metrics_dicts.append(graph_metrics)
        print(graph_metrics)

    #make a dataframe
    graph_metrics_df = pd.DataFrame(graph_metrics_dicts)

    #arrange the dataframe columns
    graph_metrics_df = graph_metrics_df[['min_edge_weight', 'node_count', 'edge_count', 'is_DAG', 'SCC']]

    #split the SCC column across different rows
    graph_metrics_df = graph_metrics_df.explode('SCC')

    #convert list to string
    graph_metrics_df['SCC'] = graph_metrics_df['SCC'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))

    #save the dataframe
    graph_metrics_df.to_csv(output_path, sep =';', index = False)


if __name__ == "__main__":

    #specify path to data and load it
    print('load data')
    embassy_df = pd.read_csv("/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv") 

    #format the dataframe
    print('format dataframe')
    embassy_df = format_df(embassy_df)

    #load list of embassy channels
    embassy_channels = open('outputs/lists/channel_usernames.txt').read().splitlines()

    #construct directed graph of forwarded messages (including self loops)
    fwd_df = embassy_df[embassy_df['fwd_source'].notnull()]
    G = construct_digraph(list(fwd_df['fwd_source']), list(fwd_df['channel_username']))
    nx.write_gexf(G, 'outputs/networks/message_forwarding_graph.gexf')
    print(G)

    #plot the full graph 
    plot_message_forwarding_network(G, 'outputs/figures/appendix_full_forward_graph.pdf')

    #filter the main directed graph (remove self-loops, keep minimum edge weight)
    I = filter_graph(G, 20)
    print(get_graph_metrics(I))

    #create sankey diagram 
    print('save sankey diagram')
    plot_sankey(I, 'outputs/figures/fig5_information_flow_messages.pdf')

    #store graph metrics for graph filtered for different edge weights
    store_graph_metrics(G,'outputs/networks/network_components.csv')