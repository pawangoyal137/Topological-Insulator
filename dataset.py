import os
import sys
import pickle
from pymatgen.io.cif import CifParser


def ti():
    data_dir = "data/ti/"
    tis = []
    for filename in os.listdir(data_dir):
        name = filename.split('.')[0]
        name = name.split('-')[-1]
        tis.append(name)
    return tis

def graph(structure):

    globals_0 = [structure.lattice]

    nodes_0 = []
    for i in structure.sites:
        nodes_0.append(i)

    edges_0 = []
    senders_0 = []
    receivers_0 = []
    for i in range(structure.num_sites):
        for j in range(structure.num_sites):
            if i == j:
                continue
            edges_0.append([structure.distance_matrix[i][j]])
            senders_0.append(i)
            receivers_0.append(j)

    data_dict = {
        "globals": globals_0,
        "nodes": nodes_0,
        "edges": edges_0,
        "senders": senders_0,
        "receivers": receivers_0
    }

    return data_dict

def main(args):
    tis = ti()
    data_dir = "data/all/"
    save_file = "data/data.pkl"
    output = open(save_file, 'wb')
    for filename in os.listdir(data_dir):
        parser = CifParser(data_dir + filename, occupancy_tolerance=100)
        structure = parser.get_structures()[0]
        x = graph(structure)
        c = ''.join(structure.formula.split())
        y = 1 if c in tis else 0

        print(c)
        pickle.dump((x, y), output, pickle.HIGHEST_PROTOCOL)
    output.close()

if __name__ == '__main__':
    args = sys.argv
    main(args)