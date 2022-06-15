import torch
import torch.nn as nn
from transformers import BertForTokenClassification


class BertForTextSegmentationEmbeddings(nn.Module):
    """input sentence embeddings inferred by bottom pre-trained BERT, contruct position embeddings.
    """

    def __init__(self, config):
        super(BertForTextSegmentationEmbeddings, self).__init__()

        self.config = config
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds, position_ids=None, input_ids=None, token_type_ids=None, past_key_values_length=None):
        # Неиспользуемые аргументы нужны для совместимости с бертом от хаггинг фэйс
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = inputs_embeds.device

        assert seq_length < self.config.max_position_embeddings, \
            f"Too long sequence is passed. Maximum allowed sequence length is {self.config.max_position_embeddings}"

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertForTextSegmentation(BertForTokenClassification):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, config):
        super(BertForTextSegmentation, self).__init__(config)

        self.bert.base_model.embeddings = BertForTextSegmentationEmbeddings(config)

        self.init_weights()
