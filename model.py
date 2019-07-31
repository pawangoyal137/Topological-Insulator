import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np

GLOBAL_SIZE = 4
NODE_SIZE = 5
EDGE_SIZE = 6

def get_graph_data_dict(num_nodes, num_edges):
	return {
		"globals": np.random.rand(GLOBAL_SIZE).astype(np.float32),
		"nodes": np.random.rand(num_nodes, NODE_SIZE).astype(np.float32),
		"edges": np.random.rand(num_edges, EDGE_SIZE).astype(np.float32),
		"senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
		"receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
	}

graph_3_nodes_4_edges = get_graph_data_dict(num_nodes=3, num_edges=4)
graph_5_nodes_8_edges = get_graph_data_dict(num_nodes=5, num_edges=8)
graph_7_nodes_13_edges = get_graph_data_dict(num_nodes=7, num_edges=13)
graph_9_nodes_25_edges = get_graph_data_dict(num_nodes=9, num_edges=25)

graph_dicts = [graph_3_nodes_4_edges, graph_5_nodes_8_edges,
               graph_7_nodes_13_edges, graph_9_nodes_25_edges]


## build graph
tf.reset_default_graph()

input_graphs = gn.utils_tf.data_dicts_to_graphs_tuple(graph_dicts)

print(input_graphs.globals)


graph_network = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.Linear(output_size=EDGE_SIZE),
    node_model_fn=lambda: snt.Linear(output_size=NODE_SIZE),
    global_model_fn=lambda: snt.Linear(output_size=GLOBAL_SIZE))

num_recurrent_passes = 3
previous_graphs = input_graphs
for unused_pass in range(num_recurrent_passes):
 	previous_graphs = graph_network(previous_graphs)
output_graphs = previous_graphs


print(output_graphs.globals)


### define input/output




### session !!!
