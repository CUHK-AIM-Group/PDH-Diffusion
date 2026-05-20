from torch import nn
import torch.nn.functional as F
from .layers import HGNN_conv
import torch


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    # def forward(self, x, G):
    #     # print(G.shape)
    #     x = F.relu(self.hgc1(x, G))
    #     x = F.dropout(x, self.dropout)
    #     x = self.hgc2(x, G)
    #     return x
    
    def forward(self, x, G):
        # print(G.shape)
        bs,history_len,s=x.shape
        series_num=G.shape[-1]
        grain_num=s//series_num
        split_size=[series_num]*grain_num
    
        x_split = torch.split(x, split_size, dim=-1)

        x_output=[]

        for g_i in range(grain_num):
            # print(x_split[g_i].shape)
            # print(G[:,g_i,:,:].shape)

            x_med = F.relu(self.hgc1(x_split[g_i], G[:,g_i,:,:]))
            x_med = F.dropout(x_med, self.dropout)
            # print('x_med',x_med.shape)
            output = self.hgc2(x_med, G[:,g_i,:,:])
            x_output.append(output)

        output = torch.cat(x_output, dim=-1)
        # print(output.shape)

        return x
