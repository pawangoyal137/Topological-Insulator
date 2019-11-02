import os
import sys
import pickle
import numpy as np
from pymatgen.io.cif import CifParser


def read_ti_id():
    ti_id = {}

    f = open("data/target/case1-icsd.txt", 'r')
    tmp = f.readlines()
    ti_id[0] = set([int(i[5:-1]) for i in tmp])

    f = open("data/target/case3-icsd.txt", 'r')
    tmp = f.readlines()
    ti_id[1] = set([int(i[5:-1]) for i in tmp])

    f = open("data/target/tci-icsd.txt", 'r')
    tmp = f.readlines()
    ti_id[2] = set([int(i[5:-1]) for i in tmp])

    f = open("data/target/ti-icsd.txt", 'r')
    tmp = f.readlines()
    tmp.remove('Sum I3 Th1\n')
    ti_id[3] = set([int(i[5:-1]) for i in tmp])
    return ti_id


def cifReader(filename):
    parser = CifParser(filename, occupancy_tolerance=100)
    structure = parser.get_structures()[0]

    atoms = set()

    for at in parser.get_structures()[0]._sites:
        x = str(list(at._species._data.keys())[0])
        atoms.add(x)
    return atoms, structure


def graph(structure, atomId):
    ids = []
    pos = []
    for at in structure._sites:
        atom_name = str(list(at._species._data.keys())[0])
        id = atomId[atom_name]
        ids.append(id)
        pos.append((at._coords))

    pos = np.array(pos)
    ids = np.array(ids)
    return pos, ids
    # globals_0 = [structure.lattice]

    # nodes_0 = []
    # for i in structure.sites:
    #     nodes_0.append(i)

    # edges_0 = []
    # senders_0 = []
    # receivers_0 = []
    # for i in range(structure.num_sites):
    #     for j in range(structure.num_sites):
    #         if i == j:
    #             continue
    #         edges_0.append([structure.distance_matrix[i][j]])
    #         senders_0.append(i)
    #         receivers_0.append(j)

    # data_dict = {
    #     "globals": globals_0,
    #     "nodes": nodes_0,
    #     "edges": edges_0,
    #     "senders": senders_0,
    #     "receivers": receivers_0
    # }

    # return data_dict

def main(args):
    ti_id = read_ti_id()

    data_dir = "data/all/"
    save_file = "data/data.pkl"
    output = open(save_file, 'wb')
    count = 0
    atoms = set()
    structs = []
    xs = []
    ys = []
    for filename in os.listdir(data_dir):
        try:
            id = filename.split('.')[0]
            id = int(id.split('MyBaseFileNameCollCode')[1])

            category = -1
            for k in ti_id:
                if id in ti_id[k]:
                    category = k
                    count += 1

            if category >= 0:
                ats, struct = cifReader(data_dir + filename)
                atoms = atoms | ats

                structs.append(struct)
                ys.append(category)
                print(count)
        except:
            print('error: ', id)


    # build atom id dict
    atomId = {}
    i = 0
    for at in atoms:
        atomId[at] = i
        i += 1
    print(atomId)

    # build x, y data
    for i in range(len(structs)):
        xs.append(graph(structs[i], atomId))

    print(xs, ys)
    pickle.dump((atomId, xs, ys), output, pickle.HIGHEST_PROTOCOL)
    output.close()

if __name__ == '__main__':
    args = sys.argv
    main(args)