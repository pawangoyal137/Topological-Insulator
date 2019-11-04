from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

GLOBAL_SIZE = 2
NODE_SIZE = 2
EDGE_SIZE = 2

#Generates random data 
def get_graph_data_dict(num_nodes, num_edges):
  return {
      "globals": np.random.rand(GLOBAL_SIZE).astype(np.float32),
      "nodes": np.random.rand(num_nodes, NODE_SIZE).astype(np.float32),
      "edges": np.random.rand(num_edges, EDGE_SIZE).astype(np.float32),
      "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
  }
  
graph_3_nodes_2_edges = get_graph_data_dict(num_nodes=3, num_edges=2)
graph_2_nodes_1_edges = get_graph_data_dict(num_nodes=2, num_edges=1)
graph_4_nodes_3_edges = get_graph_data_dict(num_nodes=4, num_edges=3)
graph_9_nodes_25_edges = get_graph_data_dict(num_nodes=9, num_edges=25)
graph_dicts = [graph_3_nodes_2_edges, graph_2_nodes_1_edges,graph_4_nodes_3_edges,graph_9_nodes_25_edges]
graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graph_dicts)
  
  
#Prints the graph tuple (and not graph dict)
def print_graphs_tuple(graphs_tuple):
  print("Shapes of `GraphsTuple`'s fields:")
  print(graphs_tuple.map(lambda x: x if x is None else x.shape, fields=graphs.ALL_FIELDS))
  print("\nData contained in `GraphsTuple`'s fields:")
  print("globals:\n{}".format(graphs_tuple.globals))
  print("nodes:\n{}".format(graphs_tuple.nodes))
  print("edges:\n{}".format(graphs_tuple.edges))
  print("senders:\n{}".format(graphs_tuple.senders))
  print("receivers:\n{}".format(graphs_tuple.receivers))
  print("n_node:\n{}".format(graphs_tuple.n_node))
  print("n_edge:\n{}".format(graphs_tuple.n_edge))
  

 #Plot the graph_tuple
def plot_graphs_tuple_np(graphs_tuple):
  networkx_graphs = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
  num_graphs = len(networkx_graphs)
  _, axes = plt.subplots(1, num_graphs, figsize=(5*num_graphs, 5))
  if num_graphs == 1:
    axes = axes,
  for graph, ax in zip(networkx_graphs, axes):
    plot_graph_networkx(graph, ax)


def plot_graph_networkx(graph, ax, pos=None):
  node_labels = {node: "{:.3g}".format(data["features"][0])
                 for node, data in graph.nodes(data=True)
                 if data["features"] is not None}
  edge_labels = {(sender, receiver): "{:.3g}".format(data["features"][0])
                 for sender, receiver, data in graph.edges(data=True)
                 if data["features"] is not None}
  global_label = ("{:.3g}".format(graph.graph["features"][0])
                  if graph.graph["features"] is not None else None)

  if pos is None:
    pos = nx.spring_layout(graph)
  nx.draw_networkx(graph, pos, ax=ax, labels=node_labels)

  if edge_labels:
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax)

  if global_label:
    plt.text(0.05, 0.95, global_label, transform=ax.transAxes)

  ax.yaxis.set_visible(False)
  ax.xaxis.set_visible(False)
  return pos


#Uncomment below code to get a specific example
## Global features for graph 0.
#globals_0 = [1.,2.]
#
## Node features for graph 0.
#nodes_0 = [[10., 20.],  # Node 0
#           [11., 21.],  # Node 1
#           [12., 22.]]  # Node 2
#
## Edge features for graph 0.
#edges_0 = [[100., 200.],   # Edge 0
#           [101., 201.]]  # Edge 1
#
## The sender and receiver nodes associated with each edge for graph 0.
#senders_0 = [0,  # Index of the sender node for edge 0
#             1]  # Index of the sender node for edge 1
#             
#receivers_0 = [1,  # Index of the receiver node for edge 0
#               2]  # Index of the receiver node for edge 1
#
#data_dict_0 = {
#    "globals": globals_0,
#    "nodes": nodes_0,
#    "edges": edges_0,
#    "senders": senders_0,
#    "receivers": receivers_0
#}
#
#data_dict_list_temp = [data_dict_0]
