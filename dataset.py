import os
import sys
import pickle
from pymatgen.io.cif import CifParser


def read_ti_id():
    ti_id = {}

    f = open("data/target/case1-icsd.txt", 'r')
    tmp = f.readlines()
    ti_id['case1'] = [int(i[5:-1]) for i in tmp]

    f = open("data/target/case3-icsd.txt", 'r')
    tmp = f.readlines()
    ti_id['case3'] = [int(i[5:-1]) for i in tmp]

    f = open("data/target/tci-icsd.txt", 'r')
    tmp = f.readlines()
    ti_id['tci'] = [int(i[5:-1]) for i in tmp]

    f = open("data/target/ti-icsd.txt", 'r')
    tmp = f.readlines()
    tmp.remove('Sum I3 Th1\n')
    ti_id['ti'] = [int(i[5:-1]) for i in tmp]
    return ti_id


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
    ti_id = read_ti_id()

    # for k in ti_id:
    #     print(k)
    #     print(len(ti_id[k]))


    data_dir = "data/all/"
    save_file = "data/data.pkl"
    output = open(save_file, 'wb')
    for filename in os.listdir(data_dir):
        try:
            parser = CifParser(data_dir + filename, occupancy_tolerance=100)
            structure = parser.get_structures()[0]

            id = filename.split('.')[0]
            id = int(id.split('MyBaseFileNameCollCode')[1])

            y = 'Others'
            x = structure
            for k in ti_id:
                if id in ti_id[k]:
                    y = k
                    break
        except:
            print("A bad file")

        pickle.dump((id, x, y), output, pickle.HIGHEST_PROTOCOL)
    output.close()

if __name__ == '__main__':
    args = sys.argv
    main(args)