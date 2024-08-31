'''
This script plots the information flow between channels based on forwarded messages
'''

#import libraries
import pandas as pd
import networkx as nx
from ast import literal_eval
import plotly.graph_objects as go

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


#import libraries
import networkx as nx
import pandas as pd
from ast import literal_eval


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
    filter graph G to retain a DAG (for the present dataset, this is achieved if n > 20)
    '''
    print('remove self-loops')
    G.remove_edges_from(nx.selfloop_edges(G))

    #filter the graph by edge weights until it we retain a DAG
    n = min_edge_weight
    print('filter by edge weight ', str(n))
    H = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data['weight'] > n:
            H.add_edge(u, v, weight=data['weight'])

    print('is DAG:', nx.is_directed_acyclic_graph(H))
    return H 


if __name__ == "__main__":

    #load datasets
    print('load data')
    embassy_df = pd.read_csv("/home/tom/Documents/data/geopolitics_of_propaganda/4cat_data_sample.csv") 

    #format the dataframe
    print('format dataframe')
    embassy_df = format_df(embassy_df)

    #construct directed graph of forwarded messages
    print('get message forwarding network')
    fwd_df = embassy_df[embassy_df['fwd_source'].notnull()]
    H = construct_digraph(list(fwd_df['fwd_source']), list(fwd_df['channel_username']))
    print(H)
    nx.write_gexf(H, 'outputs/networks/message_forwarding_graph.gexf')

    #filter graph and plot information flow
    print('save sankey diagram')
    I = filter_graph(H, 20)
    plot_sankey(I, 'outputs/figures/fig5_information_flow_messages.png')