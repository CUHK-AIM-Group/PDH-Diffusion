# from datasets import load_ft
import hypergraph.hypergraph_utils as hgut
import numpy as np


def load_feature_construct_H(data_dir,
                             m_prob=1.0,
                             K_neigs=[10],
                             is_probH=True,
                             split_diff_scale=False,
                             use_mvcnn_feature=False,
                             use_gvcnn_feature=True,):
    """

    :param data_dir: directory of feature data
    :param m_prob: parameter in hypergraph incidence matrix construction
    :param K_neigs: the number of neighbor expansion
    :param is_probH: probability Vertex-Edge matrix or binary
    :param use_mvcnn_feature:
    :param use_gvcnn_feature:
    :param use_mvcnn_feature_for_structure:
    :param use_gvcnn_feature_for_structure:
    :return:
    """
    # init feature

    fts_list=[]
    corr_list=[]

    for dataDict in data_dir:
        fts_list.append(dataDict['target'])
        corr_list.append(dataDict['corr'])

    # fts=np.array(fts_list)
    # print(fts.shape) #(696, 116, 128)

    H_list=[]

    for index, fts in enumerate(fts_list):

        # construct feature matrix
        fts_n = None
        fts = hgut.feature_concat(fts_n, fts)
        if fts is None:
            raise Exception(f'None feature used for model!')

        # construct hypergraph incidence matrix
        # print('Constructing hypergraph incidence matrix! \n(It may take several minutes! Please wait patiently!)')
        H = None
        # print(corr_list[index])
        corr=1-corr_list[index]
        # print(corr)
        # raise ValueError("Here terminated.")
        tmp = hgut.construct_H_with_KNN(fts,corr, K_neigs=K_neigs,
                                        split_diff_scale=split_diff_scale,
                                        is_probH=is_probH, m_prob=m_prob)
        H = hgut.hyperedge_concat(H, tmp)

        # H = hgut.hyperedge_concat(H, corr_list[index])

        if H is None:
            raise Exception('None feature to construct hypergraph incidence matrix!')
        H_list.append(H)

    # return fts, lbls, idx_train, idx_test, H
    return fts_list, H_list
