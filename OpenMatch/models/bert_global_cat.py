from re import S
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel


class BertGlobalCat(nn.Module):
    def __init__(
        self,
        pretrained: str,
        mode: str = 'cls',
        task: str = 'ranking',
        batch_size: int = 8,
        inf: bool = False
    ) -> None:
        super(BertGlobalCat, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task
        self._nins = batch_size
        self._inf = inf

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModel.from_pretrained(self._pretrained, config=self._config) ### embedding, encoding
        self._project = nn.Linear(self._config.hidden_size, self._config.hidden_size) # 768 * 768
 
        ###
        if self._task == 'global' or self._task == 'global_cat':
            self._dense = nn.Linear(self._config.hidden_size * 2, 1) ###
        ###
        else:
            raise ValueError('Task must be `ranking` or `classification`.')

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor = None, segment_ids: torch.Tensor = None, output2_cls: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._inf:

            cls, pool = self._model(input_ids.squeeze(), attention_mask = input_mask.squeeze(), token_type_ids = segment_ids.squeeze())

            return cls, pool

        else:

            output1_cls, _ = self._model(input_ids.squeeze(), attention_mask = input_mask.squeeze(), token_type_ids = segment_ids.squeeze()) #  10 * 768

            logits1 = output1_cls[:, 0, :]
            logits2 = output2_cls[:, 0, :]

            logits = torch.cat((logits1, logits2), dim=0)

            logits_attention = self._project(logits) # 10 * 768
            logits_attention = torch.mm(logits_attention, logits.t()) # 10 * 10
            logits_attention = F.softmax(logits_attention, dim=1) # 10 * 10
            logits_attention = torch.mm(logits_attention, logits) # 10 * 768

            logits = torch.cat((logits, logits_attention), dim=1) # 10 * 1536

            score = self._dense(logits).squeeze(-1)
            return score, logits