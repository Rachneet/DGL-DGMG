import torch
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from matplotlib import rc
import pickle
plt.rcParams['animation.html'] = 'html5'
plt.rcParams['figure.figsize'] = (8.0, 6.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if __name__ == '__main__':
    # pre-trained model saved with path ./model.pth
    for k in range(10):
        graph_pred_list=[]
        for i in range(1500):
            model = torch.load('./model_community.pth')
            model.eval()
            g = model()

            src_list = g.edges()[1]
            dest_list = g.edges()[0]
            # print(len(set(src_list)))
            # print(len(set(dest_list)))
            # print(src_list)
            # print(dest_list)
            evolution = []

            nx_g = nx.Graph()
            evolution.append(deepcopy(nx_g))

            for i in range(0, len(src_list), 2):
                src = src_list[i].item()
                dest = dest_list[i].item()
                if src not in nx_g.nodes():
                    nx_g.add_node(src)
                    evolution.append(deepcopy(nx_g))
                if dest not in nx_g.nodes():
                    nx_g.add_node(dest)
                    evolution.append(deepcopy(nx_g))
                nx_g.add_edges_from([(src, dest), (dest, src)])
                evolution.append(deepcopy(nx_g))

            if(len(nx_g.nodes)) !=0:
                graph_pred_list.append(nx_g)

        print(len(graph_pred_list))
        with open("./graph_pred_community_dgmg_new_"+str(k)+".dat", "wb") as f:
            pickle.dump(graph_pred_list, f)

    # with open("./graph_pred_dgmg.dat", "rb") as f:
    #     g = pickle.load( f)
    # g_list=[]
    # for graph in g:
    #     if len(graph.nodes())==0:
    #         g_list.append(graph)
    #
    # print(len(g_list))
    #
    # with open("./graph_pred_dgmg.dat", "wb") as f:
    #     pickle.dump(g_list, f)


    # with open("./graph_pred_dgmg.dat", "rb") as f:
    #     g = pickle.load(f)
    #
    # # for i in range(len(g)):
    # #     if len(g[i].nodes) !=0:
    # #         print(g[i].nodes)
    # #         g[i] = max(nx.connected_component_subgraphs(g[i]), key=len)
    # #
    # #         g[i] = nx.convert_node_labels_to_integers(g[i])
    # nx.draw(g[2], with_labels=True)
    # plt.show()

    # def pick_connected_component_new(G):
    #     # print('in pick connected component new')
    #     # print(G.number_of_nodes())
    #     # print(type(G))
    #     # adj_list = G.adjacency_list()
    #     # print(adj_list)
    #     # print(len(adj_list))
    #     for id, adj in G.adjacency():
    #         print('id : adj:', id, adj)
    #         # print('in for of pcc')
    #         if len(adj) == 0:
    #             id_min = 0
    #         else:
    #             id_min = min(adj)
    #         print('id_min: ', id_min)
    #         if id < id_min and id >= 1:
    #             # if id<id_min and id>=4:
    #             break
    #     node_list = list(range(id))  # only include node prior than node "id"
    #     # print(type(node_list))
    #     G = G.subgraph(node_list)
    #     G = max(nx.connected_component_subgraphs(G), key=len)
    #     return G
    #
    # with open("/home/rachneet/PycharmProjects/graph_generation/graphs/GraphRNN_RNN_barabasi_small_4_64_pred_100_1.dat", "rb") as f:
    #     g = pickle.load(f)
    #
    # for i in range(len(g)):
    #     g[i] = pick_connected_component_new(g[i])




    # print(evolution)
    # nx.draw(evolution[-1], with_labels=True)
    # plt.show()
    #
    # with open("./barabasi.p",'rb') as f:
    #     b = pickle.load(f)
    # print(len(b))
    #
    # def animate(i):
    #     ax.cla()
    #     g_t = evolution[i]
    #     nx.draw_networkx(g_t, with_labels=True, ax=ax,
    #                      node_color=['#FEBD69'] * g_t.number_of_nodes())
    #
    #
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    #
    # fig, ax = plt.subplots()
    # ani = animation.FuncAnimation(fig, animate,
    #                               frames=len(evolution),
    #                               interval=1200)
    #
    # # ani.save('barabasi_prediction.mp4', writer=writer)
    # plt.show()

