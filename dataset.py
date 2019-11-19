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


def cifReader(filename, atomId):
    parser = CifParser(filename, occupancy_tolerance=100)
    structure = parser.get_structures()[0]

    graph = {}
    graph['g'] = sg(structure).get_space_group_number()
    graph['lattice'] = structure._lattice._matrix
    graph['atoms'] = []
    graph['coords'] = []
    graph['name'] = filename

    for at in parser.get_structures()[0]._sites:
        x = str(list(at._species._data.keys())[0])
        graph['atoms'].append(atomId[x])
        graph['coords'].append(at._coords)
    return graph


def atom():
    ti_id = read_ti_id()

    data_dir = "data/all/"

    # build atom dict first
    atoms = set()
    for filename in os.listdir(data_dir):
        id = filename.split('.')[0]
        id_num = int(id.split('MyBaseFileNameCollCode')[1])

        # skip those are unknown
        flag = True
        for k in ti_id:
            if id_num in ti_id[k]:
                flag = False
        if flag:
            continue

        try:
            parser = CifParser(data_dir + filename, occupancy_tolerance=100)
            structure = parser.get_structures()[0]
            for at in parser.get_structures()[0]._sites:
                x = str(list(at._species._data.keys())[0])
                atoms.add(x)
        except:
            print(filename)
            
        print(len(atoms))

    atomId = {}
    i = 0
    for at in atoms:
        atomId[at] = i
        i += 1
    print(atomId)
    atom_file = "data/atom.pkl"
    output = open(atom_file, 'wb')
    pickle.dump(atomId, output, pickle.HIGHEST_PROTOCOL)
    output.close()


def main():
    ti_id = read_ti_id()

    data_dir = "data/all/"
    with open('data/atom.pkl', 'rb') as f:
        atomId = pickle.load(f)

    train_dicts = []
    test_dicts = []

    for filename in os.listdir(data_dir):
        id = filename.split('.')[0]
        id_num = int(id.split('MyBaseFileNameCollCode')[1])

        category = -1
        for k in ti_id:
            if id_num in ti_id[k]:
                category = k
        if category == -1:
            continue

        print(filename)

        try:
            graph = cifReader(data_dir + filename, atomId)
            graph['y'] = category
            if graph['g'] in typeII:
                test_dicts.append(graph)
            else:
                train_dicts.append(graph)
        except:
            print(filename)            

    train_file = "data/train.pkl"
    output = open(train_file, 'wb')
    pickle.dump(train_dicts, output, pickle.HIGHEST_PROTOCOL)
    output.close()

    test_file = "data/predict.pkl"
    output = open(test_file, 'wb')
    pickle.dump(test_dicts, output, pickle.HIGHEST_PROTOCOL)
    output.close()


if __name__ == '__main__':
    main()