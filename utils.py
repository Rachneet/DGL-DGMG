import torch
import random
import datetime
import os
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from pprint import pprint
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from datacreator import generate_dataset, generate_data,generate_data_community

def pick_connected_component_new(G):
    # print('in pick connected component new')
    # print(G.number_of_nodes())
    # print(type(G))
    # adj_list = G.adjacency_list()
    # print(adj_list)
    # print(len(adj_list))
    for id,adj in G.adjacency():
        # print('id : adj:', id,adj)
        if len(adj) == 0:
            id_min = 0
        else:
            id_min = min(adj)
        # print('id_min: ', id_min)
        if id<id_min and id>=1:
        # if id<id_min and id>=4:
            break
    node_list = list(range(id)) # only include node prior than node "id"
    # print(type(node_list))
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


def load_graph_list(fname,is_real=True):
    # print('in load graph list')
    # print(fname)
    with open(fname, "rb") as file:
        # print("in file open")
        graph_list = pkl.load(file)
        #print(graph_list)
    for i in range(len(graph_list)):
        # print('in for')
        # print(type(graph_list[i]))
        edges_with_selfloops = list(graph_list[i].selfloop_edges())
        # print(len(edges_with_selfloops))

        if len(edges_with_selfloops)>0:
            print('pass 1')
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            # print('is real')
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])

    return graph_list


#functions for the model

def weights_init(m):

    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)



def dgmg_message_weight_init(m):
    """
    This is similar as the function above where we initialize linear layers from a normal distribution with std
    1./10 as suggested by the author. This should only be used for the message passing functions, i.e. fe's in the
    paper.
    """

    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1. / 10)
            init.normal_(m.bias.data, std=1. / 10)
        else:
            raise ValueError('Expected the input to be of type nn.Linear!')

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(_weight_init)
    else:
        m.apply(_weight_init)


def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise


def date_filename(base_dir='./'):
    dt = datetime.datetime.now()
    return os.path.join(base_dir, '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second
    ))


def setup_log_dir(opts):
    log_dir = '{}'.format(date_filename(opts['log_dir']))
    mkdir_p(log_dir)
    return log_dir


def save_arg_dict(opts, filename='settings.txt'):
    def _format_value(v):
        if isinstance(v, float):
            return '{:.4f}'.format(v)
        elif isinstance(v, int):
            return '{:d}'.format(v)
        else:
            return '{}'.format(v)

    save_path = os.path.join(opts['log_dir'], filename)
    with open(save_path, 'w') as f:
        for key, value in opts.items():
            f.write('{}\t{}\n'.format(key, _format_value(value)))
    print('Saved settings to {}'.format(save_path))


def setup(args):
    opts = args.__dict__.copy()

    cudnn.benchmark = False
    cudnn.deterministic = True

    # Seed
    if opts['seed'] is None:
        opts['seed'] = random.randint(1, 10000)
    random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])

    # Dataset
    from configure import dataset_based_configure
    opts = dataset_based_configure(opts)

    assert opts['path_to_dataset'] is not None, 'Expect path to dataset to be set.'
    if not os.path.exists(opts['path_to_dataset']):
        if opts['dataset'] == 'cycles':

            generate_dataset(opts['min_size'], opts['max_size'], opts['ds_size'],
                            opts['path_to_dataset'])

        elif opts['dataset'] == 'barabasi':
            generate_data(opts['min_size'], opts['max_size'],
                          opts['path_to_dataset'])

        elif opts['dataset'] == 'community':
            generate_data_community(opts['path_to_dataset'])
        else:
            raise ValueError('Unsupported dataset: {}'.format(opts['dataset']))

    # Optimization
    if opts['clip_grad']:
        assert opts['clip_grad'] is not None, 'Expect the gradient norm constraint to be set.'

    # Log
    print('Prepare logging directory...')
    log_dir = setup_log_dir(opts)
    opts['log_dir'] = log_dir
    mkdir_p(log_dir + '/samples')

    plt.switch_backend('Agg')

    save_arg_dict(opts)
    pprint(opts)

    return opts

# index file parser
def parse_index_file(filename):
    index=[]
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load citeseer, cora and pubmed
def graph_load(dataset='cora'):

    names= ['x', 'tx', 'allx', 'graph']
    objects= []

    for i in range(len(names)):
        load = pkl.load(open('dataset/ind.{}.{}'.format(dataset,names[i]), 'rb'), encoding='latin1')
        objects.append(load)

    x,tx,allx,graph = tuple(objects)
    print(x.shape, tx.shape, allx.shape)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    print(G.number_of_edges(), G.number_of_nodes())
    adj = nx.adjacency_matrix(G)
    return adj, features, G

def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pkl.dump(G_list, f)


def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    #print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G