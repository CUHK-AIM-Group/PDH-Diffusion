from torch.nn.modules import loss
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated
from utils import weighted_average, MeanScaler, NOPScaler
# from module import GaussianDiffusion,DiffusionOutput
from pdhd_module import GaussianDiffusion, DiffusionOutput,Fractal_distribution_MLP,default
from epsilon_theta import EpsilonTheta
from hypergraph.HGNN import HGNN
from torch.nn import functional as F

import numpy as np
import scipy.sparse as sp
import dhg
import time
import torch.nn as nn


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128): #hidden_size=128
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w.squeeze(), dim=-1) #(batch_size,time_step,area_num)
        output=torch.matmul(beta.unsqueeze(1), z).squeeze() 
        return output

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def build_adj_full_d(t=4, p=4, d=1):
    rows = []
    cols = []
    for dd in range(d):
        for j in range(t-dd-1):
            for i in range(p):
                rows += [i+j*p for k in range(p)]
                cols += range((j+1+dd)*p, (j+1+dd)*p+p)
    data = np.ones(len(rows))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(t*p, t*p), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #print(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj



class NearestConvolution(nn.Module):
    """
    Use both neighbors on graph structures and neighbors of nearest distance on embedding space
    """
    def __init__(self, dim_in, dim_out):
        super(NearestConvolution, self).__init__()

        self.kn = 3
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=False)
        self.dropout = nn.Dropout(p=0.1)

        self.trans = ConvMapping(self.dim_in, self.kn)

    def _nearest_select(self, feats):
        b = feats.size()[0]
        N = feats.size()[1]
        dis = NearestConvolution.cos_dis(feats)
        _, idx = torch.topk(dis, self.kn, dim=2)
        #k_nearest = torch.stack([feats[idx[i]] for i in range(N)], dim=0)
        k_nearest = torch.stack([torch.stack([feats[j, idx[j, i]] for i in range(N)], dim=0) for j in range(b)], dim=0)                                        # (b, N, self.kn, d)
        return k_nearest

    @staticmethod
    def cos_dis(X):
        """
        cosine distance
        :param X: (b, N, d)
        :return: (b, N, N)
        """
        X = nn.functional.normalize(X, dim=-1, p=2)
        XT = X.transpose(2, 3)                             #(b, d, N)
        # return torch.bmm(X, XT)                            #(b, N, N)
        return torch.matmul(X, XT)
        # return torch.matmul(XT, X)

    def forward(self, feats, edge_dict):
        """
        :param feats:
        :param edge_dict:
        :return:
        """
        x = feats                                           # (b, N, d)
        x1 = self._nearest_select(x)                        # (b, N, kn, d)
        x_list = []
        for i in range(x1.shape[0]):
            x = self.trans(x1[i])                                  # (N, d)
            x = F.relu(self.fc(self.dropout(x)))       # (N, d')
            x_list.append(x)
        x = torch.stack(x_list, dim=0)                      #(b, N, d')
        return x

class BatchedGraphSAGEDynamicRangeMean1(nn.Module):
    def __init__(self, infeat, outfeat, use_bn=False, mean=False, add_self=False):
        super(BatchedGraphSAGEDynamicRangeMean1, self).__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.mean = mean
        self.aggregator = True
        print(infeat,outfeat)
        self.W_x = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_x.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib.weight, gain=nn.init.calculate_gain('relu'))

        self.W_neib_area = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W_neib_area.weight, gain=nn.init.calculate_gain('relu'))

        self.W_weight=nn.Linear(infeat*2, infeat, bias=True)

        # if self.use_bn:
        #     # self.bn = nn.BatchNorm1d(2*outfeat)
        #     self.bn = nn.BatchNorm1d(80)

        self.kn = 2

        self.semantic_attention = SemanticAttention(in_size=infeat*3)


    def forward(self, x, adj, p, t):

        # x: (b, N, d)
        b = x.size()[0]
        N = x.size()[1]

        series_num=116
        embedding_dim=4

        x=x.view(b,N,series_num,embedding_dim).cuda()

        start=time.time()


        #脑区之间的图与超图

        x_graph=x.permute(0,2,1,3).cuda()
        adj=adj.to('cuda')

        adj_flat=adj.view(adj.shape[0],-1)

        adj_threshold=torch.quantile(adj_flat, 0.95, dim=1, keepdim=False)
        # print(adj_threshold)

        edge_dense_list=[]
        edge_list=[]

        for bi in range(adj.shape[0]):
            edge_dense_from_adj = torch.where(adj[bi] >= adj_threshold[bi], torch.tensor(1), torch.tensor(0))
            edge_from_adj=torch.nonzero(adj[bi] >= adj_threshold[bi])
            edge_dense_list.append(edge_dense_from_adj)
            edge_list.append(edge_from_adj)
        # print(edge_list[0].shape)
        
        # hg = dhg.Hypergraph.from_feature_kNN(x_graph, k=3)
        g_list=[]
        hg_list=[]
        x_after_graph_list=[]
        for bi in range(adj.shape[0]):
            g = dhg.Graph(adj.shape[1], edge_list[bi].cpu(),merge_op="mean")
            g_list.append(g)
            hg = dhg.Hypergraph.from_graph_kHop(g, k=2)
            hg_list.append(hg)
            [a,ti,d]=x_graph[bi].shape
            Y_after_graph= hg.v2e(x_graph[bi].contiguous().view(a,-1), aggr="mean")
            x_after_graph_temp = hg.e2v(Y_after_graph, aggr="mean")
            x_after_graph=x_after_graph_temp.view(a,ti,d).cuda()
            x_after_graph_list.append(x_after_graph)
        X_after_graph=torch.stack(x_after_graph_list,dim=0) #torch.Size([8, 116, 80, 64])
        X_after_graph=X_after_graph.permute(0,2,1,3)
        end=time.time()
        du=end-start
        end=time.time()
        du=end-start

        output_x=self.W_x(x.cuda())
        # output_x_neib=self.W_neib(x_neib.cuda())
        output_x_neib_area=self.W_neib_area(X_after_graph.cuda())
        h_k=torch.cat((output_x,output_x_neib_area),dim=3)
        #linear学习权重

        h_k=self.W_weight(h_k)
        h_k = F.relu(h_k).cuda()
        if self.use_bn:
            self.bn = nn.BatchNorm2d(h_k.size(1)).to('cuda')
            h_k = self.bn(h_k)
        b_h,t_h,a_h,d_h=h_k.shape
        h_k=h_k.view(b_h,t_h,-1)
        return h_k


class MambaState(nn.Module):
    def __init__(self, d_input, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.in_proj = nn.Linear(d_input, d_model, bias=False)
    
    def forward(self, x, begin_state):
        x = self.in_proj(x)
        x = self.mamba(x)
        end_state = torch.zeros([2,2]).cuda()

        return x, end_state


class DissimilarityLoss(nn.Module):
    def __init__(self):
        super(DissimilarityLoss, self).__init__()

    def forward(self, tensor1, tensor2):
        # Normalize the tensors
        tensor1 = F.normalize(tensor1, p=1, dim=-1)
        tensor2 = F.normalize(tensor2, p=1, dim=-1)
        
        # Calculate cosine similarity
        cosine_similarity = torch.sum(tensor1 * tensor2, dim=-1)
        
        # Calculate dissimilarity (1 - cosine similarity)
        dissimilarity = 1 - cosine_similarity
        
        # Return the mean dissimilarity as the loss
        loss = dissimilarity.mean()
        return loss



class pdhdTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,  # imput size
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        dropout_rate: float,
        lags_seq: List[int],  # lag [1,24,168]
        target_dim: int,  # target dim 1
        num_gran: int,  # the number of granularities
        conditioning_length: int,
        share_ratio_list: List[float],  # betas are shared
        loss_type: str,  # L1 loss or L2 loss
        # beta_end: float,  # beta_end 0.1
        # diff_steps: int,  # diffusion steps 100
        beta_end: List[float],  # beta_end 0.1
        diff_steps: List[int],  # diffusion steps 100
        beta_schedule: str,  # linear or cosine
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        weights: List[float] = [0.8, 0.2],
        scaling: bool = True,
        share_hidden: bool = True,
        num_parallel_samples: int = 1,
        loss_weight_list: List[float]=[1,0.1],
        fractal_condition_weight:float=1,
        diffusion_condition_weight:float=1,
        use_hgnn: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.target_dim = target_dim
        self.target_dim_2 = 2*target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        print('prediction_length',prediction_length,'context_length',context_length,'history_length',history_length)
        self.length_channel=prediction_length+context_length
        self.scaling = scaling
        self.share_hidden = share_hidden
        self.weights = weights
        self.share_ratio_list = share_ratio_list
        self.num_gran = num_gran
        self.split_size = [self.target_dim]*self.num_gran

        self.loss_weight_list = loss_weight_list
        self.fractal_condition_weight = fractal_condition_weight
        self.diffusion_condition_weight = diffusion_condition_weight
        self.use_hgnn = use_hgnn
        # Multi-task loss schedule:
        # lambda(t) linearly decays from start to end over warmup epochs.
        # Total loss uses EMA-normalized task losses to avoid scale mismatch.
        self.mt_lambda_start = float(loss_weight_list[0]) if len(loss_weight_list) > 0 else 0.3
        self.mt_lambda_end = float(loss_weight_list[1]) if len(loss_weight_list) > 1 else 0.05
        self.mt_lambda_start = float(min(max(self.mt_lambda_start, 0.0), 1.0))
        self.mt_lambda_end = float(min(max(self.mt_lambda_end, 0.0), 1.0))
        self.mt_warmup_epochs = 30
        self.mt_ema_decay = 0.98
        self.register_buffer("ema_loss_frac", torch.tensor(1.0))
        self.register_buffer("ema_loss_diff", torch.tensor(1.0))
        self._mt_ema_initialized = False
        self.last_mt_stats = {}

        assert len(set(lags_seq)) == len(
            lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        self.cell_type = cell_type

        self.hidden_dim = self.target_dim
        embedding_dim=4

        self.graph_layer = nn.ModuleList([
            BatchedGraphSAGEDynamicRangeMean1(infeat=embedding_dim, outfeat=embedding_dim),
        ])

        fractal_length=40


        self.hgnn_layer=HGNN(in_ch=self.target_dim,n_class=self.target_dim,n_hid=self.target_dim,dropout=0.05)

        self.fractal_layer=Fractal_distribution_MLP(input_size=num_cells,output_size1=2,output_size2=fractal_length,
                                                    series_num=self.target_dim,length_channel=self.length_channel)
        
        self.fractal_layer_output=Fractal_distribution_MLP(input_size=num_cells,output_size1=2,output_size2=fractal_length,
                                            series_num=self.target_dim,length_channel=1)

        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]  # rnn class

        self.rnn = nn.ModuleList([rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        ) for _ in range(self.num_gran)])  # shape: (batch_size, seq_len, num_cells)

        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
        )  # dinosing network

        # print('beta_end',beta_end)

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            # share ratio, new argument to control diffusion and sampling
            share_ratio_list=share_ratio_list,
            beta_schedule=beta_schedule,
        )  # diffusion network

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )  # distribution output

        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)  # projection distribution arguments

        self.proj_fractal_args= nn.Linear(target_dim*2*fractal_length, conditioning_length*self.length_channel)

        self.proj_fractal_args_output= nn.Linear(target_dim*2*fractal_length, conditioning_length)

        self.proj_diffusion_args= nn.Linear(target_dim, conditioning_length)

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        """
        Returns lagged subsequences of a given sequence.
        Parameters
        ----------
        sequence
            the sequence from which lagged subsequences should be extracted.
            Shape: (N, T, C).
        sequence_length
            length of sequence in the T (time) dimension (axis = 1).
        indices
            list of lag indices to be used.
            eg: [1,24,168]
        subsequences_length
            length of the subsequences to be extracted.
        Returns
        --------
        lagged : Tensor
            a tensor of shape (N, S, C, I),
            where S = subsequences_length and I = len(indices),
            containing lagged subsequences.
            Specifically, lagged[i, :, j, k] = sequence[i, -indices[k]-S+j, :].
        """
        # we must have: history_length + begin_index >= 0
        # that is: history_length - lag_index - sequence_length >= 0
        # hence the following assert
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:         
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            # shape: (batch_size, 1, sub_seq_len, C)
            lagged_values.append(
                sequence[:, begin_index:end_index, ...].unsqueeze(1))
        # shape: (batch_size, sub_seq_len, C, I) I = len(indices)=3
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        unroll_length: int,
        gran_index: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """

        Args:
            lags (torch.Tensor): lagged sub-sequences
            scale (torch.Tensor): 归一化
            time_feat (torch.Tensor): _description_
            target_dimension_indicator (torch.Tensor): _description_
            unroll_length (int): _description_
            begin_state (Optional[Union[List[torch.Tensor], torch.Tensor]], optional): _description_. Defaults to None.

        Returns:
            Tuple[ torch.Tensor, Union[List[torch.Tensor], torch.Tensor], torch.Tensor, torch.Tensor, ]: _description_
        """

        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)
        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim) 
        )

        # (batch_size, target_dim, embed_dim)
        index_embeddings = self.embed(target_dimension_indicator)
        # print('index_embeddings',index_embeddings.shape) #torch.Size([32, 116, 1])

        # (batch_size, seq_len, target_dim * embed_dim)
        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )
        # (batch_size, sub_seq_len, input_dim)
        # inputs = torch.cat(
        #     (input_lags, repeated_index_embeddings, time_feat), dim=-1)
        inputs = torch.cat(
            (input_lags, repeated_index_embeddings), dim=-1)

        # unroll encoder
        rnn = self.rnn[gran_index]
        outputs, state = rnn(inputs, begin_state)
        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        item_id:torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Unrolls the RNN encoder over past and, if present, future data.
        Returns outputs and state of the encoder, plus the scale of
        past_target_cdf and a vector of static features that was constructed
        and fed as input to the encoder. All tensor arguments should have NTC
        layout.

        Parameters
        ----------
        past_time_feat
            Past time features (batch_size, history_length, num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        target_dimension_indicator
            Dimensionality of the time series (batch_size, target_dim)

        Returns
        -------
        outputs
            RNN outputs (batch_size, seq_len, num_cells)
        states
            RNN states. Nested list with (batch_size, num_cells) tensors with
        dimensions target_dim x num_layers x (batch_size, num_cells)
        scale
            Mean scales for the time series (batch_size, 1, target_dim)
        lags_scaled
            Scaled lags(batch_size, sub_seq_len, target_dim, num_lags)
        inputs
            inputs to the RNN

        """

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )
        # print(past_time_feat.shape)
        # print(future_time_feat.shape)

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length:, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length:, ...],
                 future_time_feat),
                dim=1,
            )

            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        # change1: split the sequence into fine and coarse-graine dataset

        sequences = torch.split(sequence, self.split_size, dim=2)


        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags = [self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        ) for sequence in sequences]
        # print(lags[0].shape) #torch.Size([64, 32, 112, 1])
        # raise ValueError("Here terminated.")

        # scale is computed on the context length last units of the past target
        # scale shape is (batch_size, 1, target_dim)
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length:, ...],
            past_observed_values[:, -self.context_length:, ...],
        )

        scales = torch.split(scale, self.split_size, dim=2)
        target_dimension_indicators = torch.split(
            target_dimension_indicator, self.split_size, dim=1)
        outputs = []
        states = []
        lags_scaled = []
        inputs = []
        for i in range(self.num_gran):
            output, state, lag_scaled, input = self.unroll(
                lags=lags[i],
                scale=scales[i],
                time_feat=time_feat,
                gran_index=i,
                # use the target_dimension_indicator 0-369
                target_dimension_indicator=target_dimension_indicators[0],
                unroll_length=subsequences_length,
                begin_state=None,
            )
            # print(input.shape) #torch.Size([8, 80, 580]) 
            # print(output.shape)#torch.Size([8, 80, 464])

            
            outputs.append(output)
            states.append(state)
            lags_scaled.append(lag_scaled)
            inputs.append(input)

        return outputs, states, scale, lags_scaled, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        """
        Returns the distribution of DeepVAR with respect to the RNN outputs.

        Parameters
        ----------
        rnn_outputs
            Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
        scale
            Mean scale for each time series (batch_size, seq_len, condition_scale)

        Returns
        -------
        distr
            Distribution instance
        distr_args
            Distribution arguments
        """
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args
    
    # def fractal_args(self, fractal_pdf: torch.Tensor):
    #         """
    #         """
    #         fractal_inputs=fractal_pdf.view(fractal_pdf.shape[0],-1,fractal_pdf.shape[-1]).permute(0,2,1)
    #         fractal_args = self.proj_fractal_args(fractal_inputs)
    #         return fractal_args


    def fractal_args(self, fractal_pdf: torch.Tensor,condition_length=100,length_channel=100):
        """
        """
        fractal_inputs=fractal_pdf.reshape(fractal_pdf.shape[0],-1) #[4, 82, 2, 40]-->
        fractal_args = self.proj_fractal_args(fractal_inputs)
        fractal_args=fractal_args.reshape(fractal_pdf.shape[0],length_channel,condition_length)
        return fractal_args
    
    def fractal_args_output(self, fractal_pdf: torch.Tensor,rnn_outputs:torch.Tensor,condition_length=100,length_channel=100):
        """
        """
        fractal_inputs=fractal_pdf.reshape(fractal_pdf.shape[0],-1) #[64, 82, 2, 40]-->
        fractal_args = self.proj_fractal_args_output(fractal_inputs)
        fractal_args=fractal_args.reshape(fractal_pdf.shape[0],length_channel,condition_length)
        return fractal_args
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(
        self,
        item_id:torch.Tensor,
        G:torch.Tensor,
        hqDq:torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,


    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """



        # raise ValueError("Here terminated.")
        # print(item_id)

        seq_len = self.context_length + self.prediction_length

        # print('G',G.shape)
        # print('past_time_feat',past_time_feat.shape) #torch.Size([64, 56, 6])
        # print('past_target_cdf',past_target_cdf.shape)#torch.Size([64, 56, 232])
        # print('future_time_feat',future_time_feat.shape) #torch.Size([64, 16, 6])
        # print('future_target_cdf',future_target_cdf.shape)#torch.Size([64, 16, 232])

        if self.use_hgnn:
            past_time_cdf_after_graph = self.hgnn_layer(past_target_cdf, G)
        else:
            past_time_cdf_after_graph = past_target_cdf

        # series_fractal_pdf=self.fractal_layer(past_target_cdf)
        # unroll the decoder in "training mode", i.e. by providing future data
        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            item_id=item_id,
            # corr=corr,
            past_time_feat=past_time_feat,
            past_target_cdf=past_time_cdf_after_graph,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        if self.use_hgnn:
            future_time_cdf_after_graph = self.hgnn_layer(future_target_cdf, G)
        else:
            future_time_cdf_after_graph = future_target_cdf


        #经过hgnn处理的past_time_cdf和future_time_cdf

        target = torch.cat(
            (past_time_cdf_after_graph[:, -self.context_length:, ...],
             future_time_cdf_after_graph),
            dim=1,
        )
        #7


        target = target/scale
        targets = torch.split(target, self.split_size, dim=2)
   
        rnn_outputs_2 = rnn_outputs  # outputs from multiple rnns (list)

        distr_args = [self.distr_args(rnn_output)
                      for rnn_output in rnn_outputs_2]
        

        series_fractal_pdf_args=[self.fractal_layer(rnn_output)
                                 for rnn_output in rnn_outputs_2] 
        
    
        fractal_args = [self.fractal_args(fractal_pdf_arg,condition_length=100,length_channel=self.length_channel)
                for fractal_pdf_arg in series_fractal_pdf_args]

        fractal_pdf=[]

        likelihoods = []

        if self.current_epoch<20:
            condition_weight=0
        else:
            condition_weight=self.fractal_condition_weight


        for ratio_index, share_ratio in enumerate(self.share_ratio_list):

            rnn_fractal_condition=distr_args[ratio_index]+condition_weight*fractal_args[ratio_index]
            rnn_condition=distr_args[ratio_index]

            #不同粒度跨层指导
            if ratio_index != len(self.share_ratio_list)-1 :

                x,x_coarser_grained,time=self.diffusion.log_prob_pre(targets[ratio_index],targets[ratio_index+1],share_ratio=share_ratio)

                diffusion_condition=self.proj_diffusion_args(targets[ratio_index+1])

                rnn_fractal_diffusion_condition=rnn_fractal_condition+self.diffusion_condition_weight*diffusion_condition

                cur_likelihood = self.diffusion.log_prob(x, rnn_fractal_diffusion_condition,time,
                                    share_ratio=share_ratio).unsqueeze(-1)
            else:
                cur_likelihood = self.diffusion.log_prob_full(targets[ratio_index], rnn_fractal_condition,
                    share_ratio=share_ratio).unsqueeze(-1)

            likelihoods.append(cur_likelihood)
        # raise ValueError("Here terminated.")



        if self.scaling:
            self.diffusion.scale = scale

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        observed_values = torch.cat(
            (
                past_observed_values[:, -self.context_length:, ...],
                future_observed_values,
            ),
            dim=1,
        )  # batch_size * seq_length * 370*2

        # mask the loss at one time step if one or more observations is missing
        # in the target dimensions (batch_size, subseq_length, 1)
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
       
        loss_diffusion = sum(
            loss_item * weight_item for loss_item, weight_item in zip(likelihoods, self.weights)
        )
        # Keep diffusion loss as a scalar to avoid shape/broadcast issues downstream.
        if isinstance(loss_diffusion, torch.Tensor) and loss_diffusion.ndim > 0:
            loss_diffusion = loss_diffusion.mean()
        
        # criterion=DissimilarityLoss()

        criterion = nn.MSELoss()

    
        loss_frac = criterion(
            torch.stack(series_fractal_pdf_args).float(),
            hqDq.permute(1, 0, 2, 3, 4).float(),
        )
        if isinstance(loss_frac, torch.Tensor) and loss_frac.ndim > 0:
            loss_frac = loss_frac.mean()
        
        # Multi-task normalized loss:
        # 1) Maintain EMA for each task loss scale
        # 2) Normalize each task by its EMA
        # 3) Blend with epoch-dependent lambda(t)
        if self.training:
            with torch.no_grad():
                cur_frac = loss_frac.detach().float().mean().clamp_min(1e-8)
                cur_diff = loss_diffusion.detach().float().mean().clamp_min(1e-8)
                if not self._mt_ema_initialized:
                    self.ema_loss_frac.copy_(cur_frac)
                    self.ema_loss_diff.copy_(cur_diff)
                    self._mt_ema_initialized = True
                else:
                    self.ema_loss_frac.mul_(self.mt_ema_decay).add_(
                        cur_frac * (1.0 - self.mt_ema_decay)
                    )
                    self.ema_loss_diff.mul_(self.mt_ema_decay).add_(
                        cur_diff * (1.0 - self.mt_ema_decay)
                    )

        norm_loss_frac = loss_frac / self.ema_loss_frac.detach().clamp_min(1e-8)
        norm_loss_diff = loss_diffusion / self.ema_loss_diff.detach().clamp_min(1e-8)

        epoch_id = float(getattr(self, "current_epoch", 0))
        warmup = float(max(self.mt_warmup_epochs, 1))
        progress = min(max(epoch_id / warmup, 0.0), 1.0)
        lambda_t = self.mt_lambda_start + (self.mt_lambda_end - self.mt_lambda_start) * progress

        total_loss = lambda_t * norm_loss_frac + (1.0 - lambda_t) * norm_loss_diff
        raw_total_loss = lambda_t * loss_frac + (1.0 - lambda_t) * loss_diffusion

        # Expose multi-task internals for trainer logging/debugging.
        with torch.no_grad():
            self.last_mt_stats = {
                "lambda_t": float(lambda_t),
                "raw_total_loss": float(raw_total_loss.detach().item()),
                "norm_loss_frac": float(norm_loss_frac.detach().item()),
                "norm_loss_diff": float(norm_loss_diff.detach().item()),
                "ema_loss_frac": float(self.ema_loss_frac.detach().item()),
                "ema_loss_diff": float(self.ema_loss_diff.detach().item()),
            }

        # Keep tuple interface for trainer compatibility.
        # Both slots point to the same total loss to avoid phase-switch jumps.
        return (total_loss, total_loss, likelihoods, distr_args, loss_frac, loss_diffusion)


class pdhdPredictionNetwork(pdhdTrainingNetwork):
    def __init__(self, num_parallel_samples: int=100, **kwargs) -> None:
        super().__init__(**kwargs)

        print("init the prediction network")
        self.num_parallel_samples = num_parallel_samples

        # for decoding the lags are shifted by one,
        # at the first time-step of the decoder a lag of one corresponds to
        # the last target value
        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
        share_ratio_list: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes sample paths by unrolling the RNN starting with a initial
        input and state.

        Parameters
        ----------
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        time_feat
            Dynamic features of future time series (batch_size, history_length,
            num_features)
        scale
            Mean scale for each time series (batch_size, 1, target_dim)
        begin_states
            List of initial states for the RNN layers (batch_size, num_cells)
        Returns
        --------
        sample_paths : Tensor
            A tensor containing sampled paths. Shape: (1, num_sample_paths,
            prediction_length, target_dim).
        """

        def repeat(tensor, dim=0):                    
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        # blows-up the dimension of each tensor to
        # batch_size * self.num_sample_paths for increasing parallelism


        past_target_cdf_list = torch.split(
            past_target_cdf, self.split_size, dim=2)
        repeated_past_target_cdf_list = [
            repeat(past_target_cdf) for past_target_cdf in past_target_cdf_list]

        repeated_time_feat = repeat(time_feat)
        scales_list = torch.split(scale, self.split_size, dim=2)
        repeated_scales_list = []
        repeated_scales_list = [repeat(scale) for scale in scales_list]

        repeated_target_dimension_indicator = repeat(
            target_dimension_indicator[:, :self.target_dim])

        if self.cell_type == "LSTM":
            # repeated_states_list = [repeat(s, dim=1) for begin_state in begin_states for s in begin_state]

            if isinstance(begin_states[0], torch.Tensor):
                repeated_states_list = [repeat(s, dim=1) for s in begin_states]
            else:
                repeated_states_list = [repeat(s, dim=1) for begin_state in begin_states for s in begin_state]
        else:
            repeated_states_list = [
                repeat(begin_state, dim=1) for begin_state in begin_states]
        # for each future time-units we draw new samples for this time-unit
        # and update the state
        future_samples_list = [[] for _ in range(self.num_gran)]
        for k in range(self.prediction_length):  # future samples from multi-gran
            for m in range(self.num_gran):
                share_ratio = self.share_ratio_list[m]
                lags = self.get_lagged_subsequences(
                    sequence=repeated_past_target_cdf_list[m],
                    sequence_length=self.history_length + k,
                    indices=self.shifted_lags,
                    subsequences_length=1,
                )
                rnn_outputs, repeated_states_list[m], _, _ = self.unroll(
                    begin_state=repeated_states_list[m],
                    lags=lags,
                    scale=repeated_scales_list[m],
                    gran_index=m,  # use rnn which corresponding gran
                    time_feat=repeated_time_feat[:, k: k + 1, ...],
                    target_dimension_indicator=repeated_target_dimension_indicator,
                    unroll_length=1,
                )
                # print('rnn_outputs',rnn_outputs.shape) #rnn_outputs torch.Size([4, 1, 328])
                distr_args = self.distr_args(rnn_outputs)
                # print('distr_args',distr_args.shape) #distr_args torch.Size([4, 1, 100])

                rnn_outputs_stack=[rnn_outputs]*self.length_channel

                rnn_outputs_stack_tensor=torch.cat(rnn_outputs_stack, dim=1)

                series_fractal_pdf_args=self.fractal_layer(rnn_outputs_stack_tensor)
        
                # print('series_fractal_pdf_args',series_fractal_pdf_args.shape) #series_fractal_pdf_args torch.Size([4, 82, 2, 40])

                fractal_args_stack = self.fractal_args(series_fractal_pdf_args,condition_length=100,length_channel=self.length_channel)

                fractal_args=torch.unsqueeze(fractal_args_stack[:,m,:],dim=1)
                condition_weight=self.fractal_condition_weight
                cond=distr_args+condition_weight*fractal_args
                new_samples = self.diffusion.sample(cond=cond,
                                                    share_ratio=share_ratio)

                new_samples *= repeated_scales_list[m]
                future_samples_list[m].append(new_samples)
                repeated_past_target_cdf_list[m] = torch.cat(
                    (repeated_past_target_cdf_list[m], new_samples), dim=1)

        # (batch_size * num_samples, prediction_length, target_dim)
        samples_list = [torch.cat(future_samples, dim=1)
                        for future_samples in future_samples_list]
        samples_reshape_list = [samples.reshape((-1, self.num_parallel_samples,
                                                self.prediction_length, self.target_dim,
                                                 )) for samples in samples_list]

        samples = torch.cat(samples_reshape_list, dim=3)
        return samples  # output multiple forecasts

    def forward(
        self,
        item_id:torch.Tensor,
        G:torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts samples given the trained DeepVAR model.
        All tensors should have NTC layout.
        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)

        Returns
        -------
        sample_paths : Tensor
            A tensor containing sampled paths (1, num_sample_paths,
            prediction_length, target_dim).

        """

        if self.use_hgnn:
            past_time_cdf_after_graph = self.hgnn_layer(past_target_cdf, G)
        else:
            past_time_cdf_after_graph = past_target_cdf

        # mark padded data as unobserved
        # (batch_size, target_dim, seq_len)
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        # unroll the decoder in "prediction mode", i.e. with past data only
        _, begin_states, scale, _, _ = self.unroll_encoder(
            item_id=item_id,
            past_time_feat=past_time_feat,
            past_target_cdf=past_time_cdf_after_graph,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )
        print(len(begin_states))

        return self.sampling_decoder(
            past_target_cdf=past_time_cdf_after_graph,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale, 
            begin_states=begin_states, # begin_states rnn_innergraph_embedding_list
            share_ratio_list=self.share_ratio_list,
        )
