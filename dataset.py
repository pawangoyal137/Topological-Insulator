import os
import sys
import pickle
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer as sg

typeI = {2, 147, 148, 201, 203, 205, 206}
typeII = {1, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
    41, 42, 43, 44, 45, 46, 75, 76, 77, 78, 79, 
    80, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 
    99, 100, 101, 102, 103, 104, 105, 106, 107, 
    108, 109, 110, 143, 144, 145, 146, 149, 150, 
    151, 152, 153, 154, 155, 156, 157, 158, 159, 
    160, 161, 168, 169, 170, 171, 172, 173, 177, 
    178, 179, 180, 181, 182, 183, 184, 185, 186, 
    195, 196, 197, 198, 199, 207, 208, 209, 210, 
    211, 212, 213, 214}


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

    space_group = sg(structure).get_space_group_number()

    atoms = set()

    for at in parser.get_structures()[0]._sites:
        x = str(list(at._species._data.keys())[0])
        atoms.add(x)
    return atoms, structure, space_group


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


    print(typeII)

    data_dir = "data/all/"
    count = 0
    atoms = set()
    structs = []
    names = []
    xs = []
    ys = []
    gs = []
    for filename in os.listdir(data_dir):
        try:
            id = filename.split('.')[0]
            id_num = int(id.split('MyBaseFileNameCollCode')[1])

            category = -1
            for k in ti_id:
                if id_num in ti_id[k]:
                    category = k
                    count += 1

            if category >= 0:
                ats, struct, space_group = cifReader(data_dir + filename)
                atoms = atoms | ats

                structs.append(struct)
                names.append(id_num)
                ys.append(category)
                gs.append(space_group)
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

    print(gs)

    trainName = []
    trainX = []
    trainY = []
    trainG = []
    testName = []
    testX = []
    testY = []
    testG = []

    for i in range(len(xs)):
        if gs[i] in typeII:
            # for prediction
            testName.append(names[i])
            testX.append(xs[i])
            testY.append(ys[i])
            testG.append(gs[i])
        else:
            trainName.append(names[i])
            trainX.append(xs[i])
            trainY.append(ys[i])
            trainG.append(gs[i])

    train_file = "data/train.pkl"
    output = open(train_file, 'wb')
    pickle.dump((atomId, trainName, trainX, trainY, trainG), output, pickle.HIGHEST_PROTOCOL)
    output.close()

    test_file = "data/test.pkl"
    output = open(test_file, 'wb')
    pickle.dump((atomId, testName, testX, testY, testG), output, pickle.HIGHEST_PROTOCOL)
    output.close()


if __name__ == '__main__':
    args = sys.argv
    main(args)