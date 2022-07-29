import argparse
import numpy as np

def preprocessing(dataset):
    if dataset == 'wikipedia':
        data_file = './wikipedia.csv'
        new_file = './Wikipedia.txt'
        pass
    elif dataset == 'reddit':
        data_file = './reddit.csv'
        new_file = './Reddit.txt'
    
    print(f'data file: {data_file}.')
    print(f'processed file: {new_file}.')
    data = np.loadtxt(data_file, skiprows=1, delimiter=',')
    # import ipdb; ipdb.set_trace()
    data = data[:, :3]
    max_u = np.max(data[:, 0])+1
    data[:, 1] = data[:, 1] + max_u
    np.savetxt(new_file, data, fmt=['%d', '%d', '%.2f'], delimiter=' ')
    print('processed data saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing')
    parser.add_argument('--dataset', type=str, choices=['wikipedia', 'reddit'], default='wikipedia', help='Dataset' )
    args = parser.parse_args()
    dataset = args.dataset # wikipedia, reddit
    preprocessing(dataset)
    