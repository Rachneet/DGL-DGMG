import time as tm
import datetime
import argparse
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import random
from torch.nn.utils import clip_grad_norm_
from args import *
from utils import *
import myset

from datacreator import *
from model import DGMG


def main(opts):
    t1 = tm.time()

    # Setup dataset and data loader
    if opts['dataset'] == 'cycles':

        dataset = CycleDataset(fname=opts['path_to_dataset'])
        evaluator = CycleModelEvaluation(v_min=opts['min_size'],
                                         v_max=opts['max_size'],
                                         dir=opts['log_dir'])
        printer = CyclePrinting(num_epochs=opts['nepochs'],
                                num_batches=opts['ds_size'] // opts['batch_size'])

    elif opts['dataset'] == 'barabasi':
        dataset = BarabasiDataset(fname=opts['path_to_dataset'])
        evaluator = ModelEvaluation(v_min=opts['min_size'],
                                         v_max=opts['max_size'],
                                         dir=opts['log_dir'])

    elif opts['dataset'] == 'community':
        dataset = CommunityDataset(fname=opts['path_to_dataset'])
        evaluator = ModelEvaluation(v_min=opts['min_size'],
                                         v_max=opts['max_size'],
                                         dir=opts['log_dir'])

    else:
        raise ValueError('Unsupported dataset: {}'.format(opts['dataset']))

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                             collate_fn=dataset.collate_single)

    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix', args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = tm.strftime("%Y-%m-%d %H:%M:%S", tm.gmtime())

    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    # if args.clean_tensorboard:
    #     if os.path.isdir("tensorboard"):
    #         shutil.rmtree("tensorboard")
    # configure("tensorboard/run" + time, flush_secs=5)

    # graphs = myset.create(args)
    #
    # # split datasets
    # random.seed(123)
    # random.shuffle(graphs)
    # graphs_len = len(graphs)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[0:int(0.2 * graphs_len)]
    #
    # # if use pre-saved graphs
    # # dir_input = "/dfs/scratch0/jiaxuany0/graphs/"
    # # fname_test = dir_input + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    # #     args.hidden_size_rnn) + '_test_' + str(0) + '.dat'
    # # graphs = load_graph_list(fname_test, is_real=True)
    # # graphs_test = graphs[int(0.8 * graphs_len):]
    # # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # # graphs_validate = graphs[int(0.2 * graphs_len):int(0.4 * graphs_len)]
    #
    # graph_validate_len = 0
    # for graph in graphs_validate:
    #     graph_validate_len += graph.number_of_nodes()
    # graph_validate_len /= len(graphs_validate)
    # print('graph_validate_len', graph_validate_len)
    #
    # graph_test_len = 0
    # for graph in graphs_test:
    #     graph_test_len += graph.number_of_nodes()
    # graph_test_len /= len(graphs_test)
    # print('graph_test_len', graph_test_len)
    #
    # args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    # max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    # min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])
    #
    # # args.max_num_node = 2000
    # # show graphs statistics
    # print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    # print('max number node: {}'.format(args.max_num_node))
    # print('max/min number edge: {}; {}'.format(max_num_edge, min_num_edge))
    # print('max previous node: {}'.format(args.max_prev_node))
    #
    # # save ground truth graphs
    # ## To get train and test set, after loading you need to manually slice
    # save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    # save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    # print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')


    # Initialize_model
    model = DGMG(v_max=opts['max_size'],
                 node_hidden_size=opts['node_hidden_size'],
                 num_prop_rounds=opts['num_propagation_rounds'])


    # Initialize optimizer
    if opts['optimizer'] == 'Adam':
        optimizer = Adam(model.parameters(), lr=opts['lr'])
    else:
        raise ValueError('Unsupported argument for the optimizer')

    t2 = tm.time()

    # Training
    model.train()
    for epoch in range(opts['nepochs']):
        batch_count = 0
        batch_loss = 0
        batch_prob = 0
        optimizer.zero_grad()

        for i, data in enumerate(data_loader):
            # print(data)
            # print(len(data))
            log_prob = model(actions=data)
            prob = log_prob.detach().exp()

            loss = - log_prob / args.batch_size
            prob_averaged = prob / args.batch_size

            loss.backward()

            batch_loss += loss.item()
            batch_prob += prob_averaged.item()
            batch_count += 1

            if batch_count % opts['batch_size'] == 0:
                print('Epoch: {} Average loss: {}'.format(epoch,batch_loss))

                if opts['clip_grad']:
                    clip_grad_norm_(model.parameters(), opts['clip_bound'])

                optimizer.step()

                batch_loss = 0
                batch_prob = 0
                optimizer.zero_grad()

    ##################################### batched training ######################################
    # for epoch in range(opts['nepochs']):
    #     batch_count = 0
    #     for batch, data in enumerate(data_loader):
    #
    #         log_prob = model_batch(batch_size=opts['batch_size'], actions=data)
    #
    #         loss = - log_prob / opts['batch_size']
    #         batch_avg_prob = (log_prob / opts['batch_size']).detach().exp()
    #         batch_avg_loss = loss.item()
    #         batch_count += 1
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         if opts['clip_grad']:
    #             clip_grad_norm_(model_batch.parameters(), opts['clip_bound'])
    #         optimizer.step()
    #
    #         if batch_count % opts['batch_size'] == 0:
    #             print('Epoch: {} Average loss: {}'.format(epoch,batch_avg_loss))
    #############################################################################################

    t3 = tm.time()

    model.eval()
    evaluator.rollout_and_examine(model, opts['num_generated_samples'])
    evaluator.write_summary()

    t4 = tm.time()

    print('It took {} to setup.'.format(datetime.timedelta(seconds=t2-t1)))
    print('It took {} to finish training.'.format(datetime.timedelta(seconds=t3-t2)))
    print('It took {} to finish evaluation.'.format(datetime.timedelta(seconds=t4-t3)))
    print('--------------------------------------------------------------------------')
    print('On average, an epoch takes {}.'.format(datetime.timedelta(
        seconds=(t3-t2) / opts['nepochs'])))

    del model.g
    # torch.save(model, './model_barabasi.pth')
    torch.save(model, './model_community.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGMG')

    # configure
    parser.add_argument('--seed', type=int, default=9284, help='random seed')

    # dataset
    parser.add_argument('--dataset', choices=['cycles','barabasi','community'], default='community',
                        help='dataset to use')
    parser.add_argument('--path-to-dataset', type=str, default='community.p',
                        help='load the dataset if it exists, '
                             'generate it and save to the path otherwise')

    # log
    parser.add_argument('--log-dir', default='./results',
                        help='folder to save info like experiment configuration '
                             'or model evaluation results')

    # optimization
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch size to use for training')
    parser.add_argument('--clip-grad', action='store_true', default=True,
                        help='gradient clipping is required to prevent gradient explosion')
    parser.add_argument('--clip-bound', type=float, default=0.25,
                        help='constraint of gradient norm for gradient clipping')

    args = parser.parse_args()
    from utils import setup
    opts = setup(args)

    main(opts)













































