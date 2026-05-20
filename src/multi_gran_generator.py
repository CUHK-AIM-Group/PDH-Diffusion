import numpy as np
from hypergraph.visual_data import load_feature_construct_H
import hypergraph.hypergraph_utils as hgut
# import utils.hypergraph_utils as hgut
import dhg

import torch


def creat_coarse_data(mg_dict, dataset_train, dataset_test):

    train_coarse_array_multi = []

    for gran in mg_dict:
        dataset_train_coarse = []
        dataDict_train_coarse = {}
        #按一定间隔求和，再重复成原先尺寸，即为平滑操作
        for dataDict in dataset_train:
            # reshape the array to have n columns
            coarse_array = []
            # print("len(dataDict['target']",len(dataDict['target']))
            for item in dataDict['target']:
                # print(item.shape) 
                max_gran = max([int(gran_cur) for gran_cur in mg_dict])
                trancated_length_train = len(item) - len(item) % max_gran #只剩可除尽的长度
                arr = item[0:trancated_length_train].reshape(-1, int(gran)) #reshape
                # print(arr.shape) #(32, 4)
                # sum every n elements of the array
                sum_arr = np.mean(arr, axis=1)
                sum_arr_align = np.repeat(sum_arr, int(gran))
                coarse_array.append(sum_arr_align)
                # print(sum_arr_align.shape)
                train_coarse_array_multi.append(sum_arr_align)

        print(len(train_coarse_array_multi))

        dataDict_train_coarse['target'] = np.array(train_coarse_array_multi) #只有这个替换了
        dataDict_train_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_train_coarse['start'] = dataDict['start']
        dataset_train_coarse.append(dataDict_train_coarse)

    # dataDict_test_coarse = {}
    # create a new dictionary for the coarse-grained dataset, should be put under the for loop
    dataset_test_coarse = []

    for dataDict in dataset_test:
        dataDict_test_coarse = {}  # create a new dictionar
        test_coarse_array_multi = []
        for gran in mg_dict:
            for item in dataDict['target']:
                # print(len(item))
                # calculate the index can be divided by gran
                gran_num = int(gran)
                cut_index = len(item) - len(item) % gran_num
                arr_before = item[:cut_index].reshape(-1, gran_num)
                arr_before_mean = np.mean(arr_before, axis=1)
                mean_arr_align = np.repeat(arr_before_mean, int(gran))
                if len(item) % gran_num != 0:
                    arr_after = item[cut_index:].reshape(
                        -1, len(item) % gran_num)
                    arr_after_mean = np.mean(arr_after, axis=1)
                    # sum every 4 elements of the array
                    mean_arr_align_after = np.repeat(
                        arr_after_mean, len(item) % gran_num)
                    mean_arr_align = np.concatenate(
                        (mean_arr_align, mean_arr_align_after), axis=0)
                # print(len(sum_arr_align))
                test_coarse_array_multi.append(mean_arr_align)
                # print(len(coarse_array))
        dataDict_test_coarse['target'] = np.array(test_coarse_array_multi)
        dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_test_coarse['start'] = dataDict['start']
        dataset_test_coarse.append(dataDict_test_coarse)
    # print(dataset.metadata.prediction_length)
    data_train = dataset_train_coarse
    data_test = dataset_test_coarse

    return data_train, data_test


def creat_coarse_data_2d(mg_dict, dataset_train, dataset_test):

    train_coarse_array_multi = []
    for dataDict in dataset_train:
        dataset_train_coarse = []
        dataDict_train_coarse = {}
        #按一定间隔求和，再重复成原先尺寸，即为平滑操作
        for gran in mg_dict:
            # reshape the array to have n columns
            coarse_array = []
            # print("len(dataDict['target']",len(dataDict['target']))
            for item in dataDict['target']:
                max_gran = max([int(gran_cur) for gran_cur in mg_dict])
                trancated_length_train = len(item) - len(item) % max_gran #只剩可除尽的长度
                arr = item[0:trancated_length_train].reshape(-1, int(gran)) #reshape
                # print(arr.shape) #(32, 4)
                # sum every n elements of the array
                sum_arr = np.mean(arr, axis=1)
                sum_arr_align = np.repeat(sum_arr, int(gran))
                coarse_array.append(sum_arr_align)
                # print(sum_arr_align.shape) #(112,)
                # train_coarse_array_multi.append(sum_arr_align)
            coarse_array=np.array(coarse_array)
            # print(coarse_array.shape) #(116,112)
            dataset_train_coarse.append(coarse_array)
        # print(len(train_coarse_array_multi))
        #train_coarse_array_multi:(len(dataset),series_num,time_step)
        dataDict_train_coarse['target'] = np.array(dataset_train_coarse) #只有这个替换了
        # dataDict_train_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_train_coarse['start'] = dataDict['start']
        dataDict_train_coarse['item_id'] = dataDict['item_id']
        dataDict_train_coarse['corr'] = dataDict['corr']
        train_coarse_array_multi.append(dataDict_train_coarse)

    # dataDict_test_coarse = {}
    # create a new dictionary for the coarse-grained dataset, should be put under the for loop
    test_coarse_array_multi = []

    for dataDict in dataset_test:
        dataDict_test_coarse = {}  # create a new dictionar
        dataset_test_coarse = []
        for gran in mg_dict:
            coarse_array = []
            for item in dataDict['target']:
                # print(len(item))
                # calculate the index can be divided by gran
                gran_num = int(gran)
                cut_index = len(item) - len(item) % gran_num
                arr_before = item[:cut_index].reshape(-1, gran_num)
                arr_before_mean = np.mean(arr_before, axis=1)
                mean_arr_align = np.repeat(arr_before_mean, int(gran))
                if len(item) % gran_num != 0:
                    arr_after = item[cut_index:].reshape(
                        -1, len(item) % gran_num)
                    arr_after_mean = np.mean(arr_after, axis=1)
                    # sum every 4 elements of the array
                    mean_arr_align_after = np.repeat(
                        arr_after_mean, len(item) % gran_num)
                    mean_arr_align = np.concatenate(
                        (mean_arr_align, mean_arr_align_after), axis=0)
                # print(len(sum_arr_align))
                coarse_array.append(mean_arr_align)
            coarse_array=np.array(coarse_array)
            dataset_test_coarse.append(coarse_array)
        # print(len(test_coarse_array_multi))
        dataDict_test_coarse['target'] = np.array(dataset_test_coarse)
        # dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_test_coarse['start'] = dataDict['start']
        dataDict_test_coarse['item_id'] = dataDict['item_id']
        dataDict_test_coarse['corr'] = dataDict['corr']
        test_coarse_array_multi.append(dataDict_test_coarse)
    # print(dataset.metadata.prediction_length)
    data_train = train_coarse_array_multi
    data_test = test_coarse_array_multi

    return data_train, data_test




def creat_coarse_data_elec(mg_dict, dataset_train, dataset_test):

    train_coarse_array_multi = []
    max_gran = max([int(cur_gran) for cur_gran in mg_dict])
    index_trun_start = 1 if max_gran != 48 else 25
    for gran in mg_dict:
        dataset_train_coarse = []
        dataset_test_coarse = []

        dataDict_train_coarse = {}
        for dataDict in dataset_train:
            # reshape the array to have n columns
            coarse_array = []
            for item in dataDict['target']:
                # print(item.shape)#(5833,)
                arr = item[index_trun_start:5833].reshape(-1, int(gran))
                # print(arr.shape)#(5832, 1)
                # sum every n elements of the array
                sum_arr = np.mean(arr, axis=1)
                sum_arr_align = np.repeat(sum_arr, int(gran))
                coarse_array.append(sum_arr_align)
                # print(sum_arr_align)
                train_coarse_array_multi.append(sum_arr_align)

            dataDict_train_coarse['target'] = np.array(
                train_coarse_array_multi)
            dataDict_train_coarse['feat_static_cat'] = dataDict['feat_static_cat']
            dataDict_train_coarse['start'] = dataDict['start']
            dataset_train_coarse.append(dataDict_train_coarse)

    dataDict_test_coarse = {}
    dataset_test_coarse = []
    for dataDict in dataset_test:
        # reshape the array to have 4 columns
        coarse_array = []
        for gran in mg_dict:
            dataDict_train_coarse = {}
            for item in dataDict['target']:
                # calculate the index can be divided by gran
                # print(item.shape) #4000-4144
                gran_num = int(gran)
                cut_index = len(item) - len(item) % gran_num
                arr_before = item[:cut_index].reshape(-1, gran_num)
                arr_before_mean = np.mean(arr_before, axis=1)
                mean_arr_align = np.repeat(arr_before_mean, int(gran))
                if len(item) % gran_num != 0:
                    arr_after = item[cut_index:].reshape(
                        -1, len(item) % gran_num)
                    arr_after_mean = np.mean(arr_after, axis=1)
                    # sum every 4 elements of the array
                    mean_arr_align_after = np.repeat(
                        arr_after_mean, len(item) % gran_num)
                    mean_arr_align = np.concatenate(
                        (mean_arr_align, mean_arr_align_after), axis=0)
                coarse_array.append(mean_arr_align)
            # print(len(coarse_array)) #740
        dataDict_test_coarse['target'] = np.array(coarse_array)
        dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
        dataDict_test_coarse['start'] = dataDict['start']
        dataset_test_coarse.append(dataDict_test_coarse)
    data_train = dataset_train_coarse
    data_test = dataset_test_coarse

    return data_train, data_test


def creat_coarse_data_from_graph(mg_dict, dataset_train, dataset_test,series_num):
    
    neigs_dict=[]

    for grain_i in mg_dict:
        if grain_i !=1:
            # neigs_dict.append(series_num//grain_i)
            neig=grain_i*2
            neigs_dict.append(neig)
            if neig>series_num:
                print('wrong mg_dict!!!')
        else:
            neigs_dict.append(grain_i)


    H_list=[]
    G_list=[]
    # # fts, H = load_feature_construct_H(dataset_train,K_neigs=[10],)
    # for grain_i in range(len(mg_dict)):
    #     fts, H = load_feature_construct_H(dataset_train,m_prob=1.0,K_neigs=neigs_dict[grain_i],)
    #     H_list.append(H)
    #     G = hgut.generate_G_from_H(H)
    #     # if(G[5].all()==H[5].all()):
    #     #     print('yes')
    #     G_list.append(G)
    # print('len(H_list)',len(H_list))
    # print(H_list[0][0].shape)

    # raise ValueError("Here terminated.")



    #生成[grain，data_num,{}]的结构

    # train_coarse_array_multi_grain = []
    # for g_i in range(len(mg_dict)):
    #     train_coarse_array_multi_data = []
    #     fts, H = load_feature_construct_H(dataset_train,m_prob=1.0,K_neigs=neigs_dict[g_i],)
    #     G = hgut.generate_G_from_H(H) #G:[train_data_num,series_num,,series_num]
    #     for i,dataDict in enumerate(dataset_train):
    #         dataset_train_coarse = []
    #         dataDict_train_coarse = {}
    #         coarse_array=dataDict['target']
    #         dataset_train_coarse.append(coarse_array)
    #         dataDict_train_coarse['target'] = np.array(dataset_train_coarse)
    #         # dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
    #         dataDict_train_coarse['start'] = dataDict['start']
    #         dataDict_train_coarse['item_id'] = dataDict['item_id']
    #         dataDict_train_coarse['G'] = G[i]
    #         train_coarse_array_multi_data.append(dataDict_train_coarse)
    #     train_coarse_array_multi_grain.append(train_coarse_array_multi_data)

    # test_coarse_array_multi_grain = []
    # for g_i in range(len(mg_dict)):
    #     test_coarse_array_multi_data = []
    #     fts, H = load_feature_construct_H(dataset_test,m_prob=1.0,K_neigs=neigs_dict[g_i],)
    #     G = hgut.generate_G_from_H(H) #G:[train_data_num,series_num,,series_num]
    #     for i,dataDict in enumerate(dataset_test):
    #         dataset_test_coarse = []
    #         dataDict_test_coarse = {}
    #         coarse_array=dataDict['target']
    #         dataset_test_coarse.append(coarse_array)
    #         dataDict_test_coarse['target'] = np.array(dataset_test_coarse)
    #         # dataDict_test_coarse['feat_static_cat'] = dataDict['feat_static_cat']
    #         dataDict_test_coarse['start'] = dataDict['start']
    #         dataDict_test_coarse['item_id'] = dataDict['item_id']
    #         dataDict_test_coarse['G'] = G[i]
    #         test_coarse_array_multi_data.append(dataDict_test_coarse)
    #     test_coarse_array_multi_grain.append(test_coarse_array_multi_data)

    # data_train = train_coarse_array_multi_grain
    # data_test = test_coarse_array_multi_grain


    #生成[data_num,{}]的结构，grain_num在grain中，即G：[grain_num,series_num,series_num]

    # H_list=[]
    G_list=[]
    for grain_i in range(len(mg_dict)):
        fts, H = load_feature_construct_H(dataset_train,m_prob=1.0,K_neigs=neigs_dict[grain_i],)
        # H_list.append(H)
        G = hgut.generate_G_from_H(H)
        G_list.append(G) #[grain_num,data_num,series_num,series_num]

    train_coarse_array_multi_data = []
    train_coarse_array_multi_grain = []
    for i,dataDict in enumerate(dataset_train):

        dataset_train_coarse = []
        dataDict_train_coarse = {}

        for g_i in range(len(mg_dict)):

            coarse_array=dataDict['target']
            dataset_train_coarse.append(coarse_array)
            dataDict_train_coarse['target'] = np.array(dataset_train_coarse)

            dataDict_train_coarse['start'] = dataDict['start']
            dataDict_train_coarse['item_id'] = dataDict['item_id']

            dataDict_train_coarse['G'] = np.array(G_list)[:,i,:,:]

        train_coarse_array_multi_data.append(dataDict_train_coarse)

    G_list=[]
    for grain_i in range(len(mg_dict)):
        fts, H = load_feature_construct_H(dataset_test,m_prob=1.0,K_neigs=neigs_dict[grain_i],)
        # H_list.append(H)
        G = hgut.generate_G_from_H(H)
        G_list.append(G) #[grain_num,data_num,series_num,series_num]

    test_coarse_array_multi_data = []
    test_coarse_array_multi_grain = []
    for i,dataDict in enumerate(dataset_test):

        dataset_test_coarse = []
        dataDict_test_coarse = {}

        for g_i in range(len(mg_dict)):

            coarse_array=dataDict['target']
            dataset_test_coarse.append(coarse_array)
            dataDict_test_coarse['target'] = np.array(dataset_test_coarse)

            dataDict_test_coarse['start'] = dataDict['start']
            dataDict_test_coarse['item_id'] = dataDict['item_id']

            dataDict_test_coarse['G'] = np.array(G_list)[:,i,:,:]

        test_coarse_array_multi_data.append(dataDict_test_coarse)

    data_train = train_coarse_array_multi_data
    data_test = test_coarse_array_multi_data


    print(len(data_train))
    print(len(data_train[0]))


    return data_train, data_test



#对target进行直接的超图聚合
def creat_coarse_data_from_graph_different(mg_dict, dataset_train, dataset_test,series_num,per):
    
    neigs_dict=[]

    for grain_i in mg_dict:
        if grain_i !=1:
            # neigs_dict.append(series_num//grain_i)
            neig=grain_i*2
            neigs_dict.append(neig)
            if neig>series_num:
                print('wrong mg_dict!!!')
        else:
            neigs_dict.append(grain_i)


    H_list=[]
    G_list=[]


    #生成[data_num,{}]的结构，grain_num在grain中，即G：[grain_num,series_num,series_num]

    # H_list=[]
    G_list=[]
    for grain_i in range(len(mg_dict)):
        fts, H = load_feature_construct_H(dataset_train,m_prob=1.0,K_neigs=neigs_dict[grain_i],)
        # H_list.append(H)
        G = hgut.generate_G_from_H(H)
        G_list.append(G) #[grain_num,data_num,series_num,series_num]

    train_coarse_array_multi_data = []
    train_coarse_array_multi_grain = []
    perctg=per
    for i,dataDict in enumerate(dataset_train):

        dataset_train_coarse = []
        dataDict_train_coarse = {}

        corr=dataDict['corr']

        # adj_flat=adj.view(adj.shape[0],-1)

        adj=torch.tensor(corr)
        # print(adj.shape)

        adj_threshold=np.quantile(corr, perctg)
        # print(adj_threshold)

        edge_dense_list=[]
        edge_list=[]


        # edge_dense_from_adj = np.where(adj[bi] >= adj_threshold[bi], 1, 0)
        # edge_from_adj=np.nonzero(adj[bi] >= adj_threshold[bi])
        edge_dense_from_adj = torch.where(adj >= adj_threshold, torch.tensor(1), torch.tensor(0))
        edge_from_adj=torch.nonzero(adj >= adj_threshold)
        # edge_dense_list.append(edge_dense_from_adj)
        # edge_list.append(edge_from_adj)
        # print(edge_from_adj.shape) #torch.Size([338, 2])
        # raise ValueError("Here terminated.")

        for g_i in range(len(mg_dict)):

            if g_i!=0:
                x_graph=dataDict['target'] #(82, 1200)
                x_graph=torch.tensor(x_graph)
                # x_graph=x.permute(0,2,1,3).cuda()
                g_list=[]
                hg_list=[]
                x_after_graph_list=[]
                [series_num,length]=x_graph.shape

                # print(edge_from_adj.shape)
                g = dhg.Graph(adj.shape[0], edge_from_adj,merge_op="mean")
                g_list.append(g)
                # print(g.e)
                # print(neigs_dict[g_i])
                hg = dhg.Hypergraph.from_graph_kHop(g, k=int(neigs_dict[g_i]))
                # print(hg.e)
                hg_list.append(hg)
                # [a,ti,d]=x_graph[bi].shape
                # print(x_graph[b].shape)
                # Y_after_graph= hg.v2e(x_graph[bi].contiguous().view(length,-1), aggr="mean")
                Y_after_graph= hg.v2e(x_graph, aggr="mean")
                x_after_graph_temp = hg.e2v(Y_after_graph, aggr="mean")
                # x_after_graph=x_after_graph_temp.view(a,ti,d).cuda()
                x_after_graph=x_after_graph_temp
                x_after_graph_list.append(x_after_graph)
                # print(Y_after_graph.shape)
                # print(x_after_graph.shape)
                X_after_graph=torch.stack(x_after_graph_list,dim=0) #torch.Size([8, 116, 80, 64])
                # X_after_graph=X_after_graph.permute(0,2,1,3)
                # print('X_after_graph',X_after_graph.shape) #X_after_graph torch.Size([1, 82, 1200])
                coarse_array=X_after_graph.squeeze()
                # raise ValueError("Here terminated.")
            else:
                coarse_array=dataDict['target']
            # print('coarse_array',coarse_array.shape) #(82, 1200)
            # print('corr',corr.shape) #(82, 82)
            # raise ValueError("Here terminated.")
            dataset_train_coarse.append(np.array(coarse_array))
            dataDict_train_coarse['target'] = np.array(dataset_train_coarse)
            # print(dataDict_train_coarse['target'].shape)

            dataDict_train_coarse['start'] = dataDict['start']
            dataDict_train_coarse['item_id'] = dataDict['item_id']

            dataDict_train_coarse['G'] = np.array(G_list)[:,i,:,:]

        train_coarse_array_multi_data.append(dataDict_train_coarse)

    G_list=[]
    for grain_i in range(len(mg_dict)):
        fts, H = load_feature_construct_H(dataset_test,m_prob=1.0,K_neigs=neigs_dict[grain_i],)
        # H_list.append(H)
        G = hgut.generate_G_from_H(H)
        G_list.append(G) #[grain_num,data_num,series_num,series_num]

    test_coarse_array_multi_data = []
    test_coarse_array_multi_grain = []
    for i,dataDict in enumerate(dataset_test):

        dataset_test_coarse = []
        dataDict_test_coarse = {}

        corr=dataDict['corr']

        # adj_flat=adj.view(adj.shape[0],-1)

        adj=torch.tensor(corr)
        # print(adj.shape)


        adj_threshold=np.quantile(corr, perctg)
        # print(adj_threshold)

        edge_dense_list=[]
        edge_list=[]


        # edge_dense_from_adj = np.where(adj[bi] >= adj_threshold[bi], 1, 0)
        # edge_from_adj=np.nonzero(adj[bi] >= adj_threshold[bi])
        edge_dense_from_adj = torch.where(adj >= adj_threshold, torch.tensor(1), torch.tensor(0))
        edge_from_adj=torch.nonzero(adj >= adj_threshold)

        for g_i in range(len(mg_dict)):

            if g_i!=0:
                x_graph=dataDict['target'] #(82, 1200)
                x_graph=torch.tensor(x_graph)
                # x_graph=x.permute(0,2,1,3).cuda()
                g_list=[]
                hg_list=[]
                x_after_graph_list=[]
                [series_num,length]=x_graph.shape

                # print(edge_from_adj.shape)
                g = dhg.Graph(adj.shape[0], edge_from_adj,merge_op="mean")
                g_list.append(g)
                # print(g.e)
                # print(neigs_dict[g_i])
                hg = dhg.Hypergraph.from_graph_kHop(g, k=int(neigs_dict[g_i]))
                # print(hg.e)
                hg_list.append(hg)

                Y_after_graph= hg.v2e(x_graph, aggr="mean")
                x_after_graph_temp = hg.e2v(Y_after_graph, aggr="mean")
                # x_after_graph=x_after_graph_temp.view(a,ti,d).cuda()
                x_after_graph=x_after_graph_temp
                x_after_graph_list.append(x_after_graph)


                X_after_graph=torch.stack(x_after_graph_list,dim=0) #torch.Size([8, 116, 80, 64])

                # print('X_after_graph',X_after_graph.shape) #X_after_graph torch.Size([1, 82, 1200])
                coarse_array=X_after_graph.squeeze()
                # raise ValueError("Here terminated.")
            else:
                coarse_array=dataDict['target']

            # coarse_array=dataDict['target']
            dataset_test_coarse.append(np.array(coarse_array))
            dataDict_test_coarse['target'] = np.array(dataset_test_coarse)

            dataDict_test_coarse['start'] = dataDict['start']
            dataDict_test_coarse['item_id'] = dataDict['item_id']

            dataDict_test_coarse['G'] = np.array(G_list)[:,i,:,:]

        test_coarse_array_multi_data.append(dataDict_test_coarse)

    data_train = train_coarse_array_multi_data
    data_test = test_coarse_array_multi_data


    print(len(data_train))
    print(len(data_train[0]))


    return data_train, data_test