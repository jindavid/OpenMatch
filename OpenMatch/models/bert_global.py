from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel

class SelfAttentionLayer(nn.Module):
    def __init__(self, nhid, nins):
        super(SelfAttentionLayer, self).__init__()
        self.nhid = nhid
        self.nins = nins
        self.project = nn.Sequential(
            # nn.Linear(nhid, 64), # H =64, 2F = nhid
            # nn.ReLU(True),
            # nn.Linear(64, 1)
            nn.Linear(nhid, 1)
        )

    def forward(self, inputs, index, claims): # input: other node
        tmp = None
        # if index > -1:
        idx = torch.LongTensor([index]).cuda()
        # print(idx)
        own = torch.index_select(inputs, 0, idx) # 1 x 768
        # print(own.size())
        own = own.repeat(self.nins, 1) # 4 x 768
        # print(own.size())
        # print(own, inputs)
        # print(inputs.size()) # 4 x 768
        
        tmp = torch.cat((own, inputs), 1) # 4 * 1536
        # print(tmp.size())
        
        # else:
        #     claims = claims.unsqueeze(1)
        #     claims = claims.repeat(1, self.nins, 1)
        #     tmp = torch.cat((claims, inputs), 2)
        # # before
        # print(self.nhid)
        attention = self.project(tmp)
        # print(attention.size())
        
        weights = F.softmax(attention.squeeze(-1), dim=0)
        # print(weights.size())
        outputs = (inputs * weights.unsqueeze(-1)).sum(dim=0)
        # print(outputs.size())
        # exit()
        
        return outputs

class AttentionLayer(nn.Module):
    def __init__(self, nins, nhid): # nin: node num
        super(AttentionLayer, self).__init__()
        self.nins = nins
        self.attentions = [SelfAttentionLayer(nhid=nhid * 2, nins=nins) for _ in range(nins)] ### * 2

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, inputs):
        # outputs = torch.cat([att(inputs) for att in self.attentions], dim=1)
        outputs = torch.cat([self.attentions[i](inputs, i, None) for i in range(self.nins)], dim=0)
        outputs = outputs.view(inputs.shape)
        return outputs

#########
# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_size):
#         super(AttentionLayer, self).__init__()

#         self._wq = nn.Linear(hidden_size, hidden_size)
#         self._wk = nn.Linear(hidden_size, hidden_size)
#         self._wv = nn.Linear(hidden_size, hidden_size)
#         self._hidden_size = hidden_size


#     def forward(self, inputs):

#         Q = self._wq(inputs)
#         K = self._wk(inputs)
#         V = self._wv(inputs)

#         attention = torch.matmul(F.softmax(Q * torch.transpose(K) /  (self._hidden_size)**0.5), V)

#         return attention
# #########

class BertGlobal(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking',
        batch_size: int = 4
    ) -> None:
        super(BertGlobal, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task
        self._nins = batch_size

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModel.from_pretrained(self._pretrained, config=self._config)
        # self._attentionlayer = AttentionLayer(self._config.hidden_size) #########
        self._attentionlayer = AttentionLayer(self._nins, self._config.hidden_size)
 
        if self._task == 'ranking':
            self._dense = nn.Linear(self._config.hidden_size, 1)
        ###
        elif self._task == 'global':
            self._dense = nn.Linear(self._config.hidden_size, 1) ### 1
        ###
        elif self._task == 'classification':
            self._dense = nn.Linear(self._config.hidden_size, 2)
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(input_ids, attention_mask = input_mask, token_type_ids = segment_ids)
        
        if self._mode == 'cls':
            logits = output[0][:, 0, :]
        elif self._mode == 'pooling':
            logits = output[1]
        else:
            raise ValueError('Mode must be `cls` or `pooling`.')

        logits = self._attentionlayer(logits) #########
        # print(logits.size())
        score = self._dense(logits).squeeze(-1)
        # print(score)
        # exit()
        # print(input_ids, logits, score) ###
        return score, logits