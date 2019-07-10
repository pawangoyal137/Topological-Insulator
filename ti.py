import pickle


def read_data():

    data = []
    filename = 'data/data.pkl'
    with open(filename, 'rb') as f:
        while 1:
            try:
                inputs = pickle.load(f)
                print(inputs[0]['globals'])
                data.append(inputs)
            except EOFError:
                break

    return data


def main():

    ## data

    ## model

    ## optimizer and init

    ## session run


if __name__ == '__main__':
    main()