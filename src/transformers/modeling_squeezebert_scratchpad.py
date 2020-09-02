from abc import ABC
import math
from copy import deepcopy

from IPython import embed
import enum
import itertools
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ModuleList, ModuleDict, Sequential

from modeling_util import gelu, swish, ACT2FN

import logging
logger = logging.getLogger(__name__)

'''

Hierarchy:
  SqueezeBertEncoder
    SqueezeBertModule
      SqueezeBertSelfAttention
        CA
        CDL


Data dimensions:
- N = batch
- C = channels (sometimes called "hidden size")
- W = seq_len (W is "width" in computer vision CNNs, so we use the same name here)

Filter dimensions:
- N = filter ID
- C = channels
- W = 1 (at least for now), meaning that filters touch one sequence element at a time


PyTorch's fully-connected layer, called Linear(), expects input data to be in NWC form.
PyTorch's Conv1d layer expects input data to be in NCW form.


'''

def transpose_x(x):
    return x.permute(0, 2, 1)  # [N, W, C] <--> {N, C, W]

class MatMulWrapper(torch.nn.Module):
    '''
    Wrapper for torch.matmul(). A bit silly, but this makes flop-counting easier to implement.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, mat1, mat2):
        '''

        :param inputs: two torch tensors
        :return: matmul of these tensors

        Here are the typical dimensions found in BERT (the B is optional)
            mat1.shape: [B, <optional extra dims>, M, K]
            mat2.shape: [B, <optional extra dims>, K, N]
            output shape: [B, <optional extra dims>, M, N]
        '''
        return torch.matmul(mat1, mat2)


class SqueezeBertLayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size, eps=1e-12):
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size, eps=eps) # instantiates self.{weight, bias, eps}

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = nn.LayerNorm.forward(self, x)
        return x.permute(0, 2, 1)


class CDL(nn.Module):
    '''
    CDL: Conv, Dropout, LayerNorm
    '''
    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()

        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.layernorm = SqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)
        x = self.dropout(x)
        x = x + input_tensor
        x = self.layernorm(x)
        return x


class CA(nn.Module):
    '''
    CA: Conv, Activation
    '''
    def __init__(self, cin, cout, groups, act):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.act = ACT2FN[act]

    def forward(self, x):
        output = self.conv1d(x)
        return self.act(output)

class SqueezeBertSelfAttention(nn.Module):
    def __init__(self, config, cin, q_groups=1, k_groups=1, v_groups=1):
        '''
        config = used for some things; ignored for others (work in progress...)
        cin = input channels = output channels
        groups = number of groups to use in conv1d layers
        '''
        super().__init__()
        if cin % config.num_attention_heads != 0:
            raise ValueError(
                "cin (%d) is not a multiple of the number of attention "
                "heads (%d)" % (cin, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(cin / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key   = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = nn.Conv1d(in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1) # this is unchanged from original BERT code, because it's applied to the output of (Q K^T V), which is in the same format here as in the original code

        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()


    def transpose_for_scores(self, x):
        '''
        input: [N, C, W]
        output: [N, C1, W, C2]
            where C1 is the head index, and C2 is one head's contents
        '''
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1]) # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2) # [N, C1, C2, W] --> [N, C1, W, C2]


    def transpose_key_for_scores(self, x):
        '''
        input: [N, C, W]
        output: [N, C1, C2, W]
            where C1 is the head index, and C2 is one head's contents
        '''
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1]) # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        # no `permute` needed
        return x

    def transpose_output(self, x):
        '''
        input: [N, C1, W, C2]
        output: [N, C, W]
        '''
        x = x.permute(0, 1, 3, 2).contiguous() # [N, C1, C2, W]
        new_x_shape = (x.size()[0], self.all_head_size, x.size()[3]) # [N, C, W]
        x = x.view(*new_x_shape)
        return x


    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        '''
        expects hidden_states in NCW form.
        '''
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = self.matmul_qk(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = self.matmul_qkv(attention_probs, value_layer)
        context_layer = self.transpose_output(context_layer)
        if output_all_encoded_layers:
            return {'attention_scores': attention_scores, 'context_layer': context_layer} # note that TinyBERT also uses attention_scores for distillation
        else:
            return context_layer


class SqueezeBertModule(nn.Module):
    def __init__(self, config, hidden_size, intermediate_size, q_groups=1, k_groups=1, v_groups=1,
                 intermediate_groups=1, post_attention_groups=1, output_groups=1):
        '''
        hidden_size = input chans = output chans for Q, K, V (they are all the same ... for now) = output chans for the module
        intermediate_size = output chans for intermediate layer
        groups = number of groups for all layers in the BertModule. (eventually we could change the interface to allow different groups for different layers)
        '''
        super().__init__()

        c0 = hidden_size
        c1 = hidden_size
        c2 = intermediate_size
        c3 = hidden_size

        self.attention = SqueezeBertSelfAttention(config=config, cin=c0, q_groups=q_groups,
                                                k_groups=k_groups, v_groups=v_groups)
        self.post_attention = CDL(cin=c0, cout=c1, groups=post_attention_groups, dropout_prob=config.hidden_dropout_prob)
        self.intermediate = CA(cin=c1, cout=c2, groups=intermediate_groups, act=config.hidden_act)
        self.output = CDL(cin=c2, cout=c3, groups=output_groups, dropout_prob=config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers):
        attention_output = self.attention(hidden_states, attention_mask, output_all_encoded_layers)
        if output_all_encoded_layers:
            attention_scores = attention_output['attention_scores']
            attention_output = attention_output['context_layer']
        post_attention_output = self.post_attention(attention_output, hidden_states)
        intermediate_output = self.intermediate(post_attention_output)
        layer_output = self.output(intermediate_output, post_attention_output)
        if output_all_encoded_layers:
            # deliberately making 'attention_score' singular, to signify that layer_output's from just 1 layer. the plural attention_scores above is legacy.
            return {'attention_score': attention_scores, 'feature_map': layer_output}
        else:
            return layer_output


class SqueezeBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        q_groups = getattr(config, 'q_groups', 1)
        k_groups = getattr(config, 'k_groups', 1)
        v_groups = getattr(config, 'v_groups', 1)
        intermediate_groups = getattr(config, 'intermediate_groups', 1)
        post_attention_groups = getattr(config, 'post_attention_groups', 1)
        output_groups = getattr(config, 'output_groups', 1)

        if hasattr(config, 'embedding_size'):
            assert config.embedding_size == hidden_size, "if you want embedding_size != intermediate hidden_size, " \
                                                         "please add a Conv1d layer beofre the first BertModule " \
                                                         "that adjusts the number of channels."

        layers = [SqueezeBertModule(config, hidden_size=hidden_size, intermediate_size=intermediate_size,
                                          q_groups=q_groups, k_groups=k_groups, v_groups=v_groups,
                                          intermediate_groups=intermediate_groups, post_attention_groups=post_attention_groups,
                                          output_groups=output_groups)
                                          for _ in range(config.num_hidden_layers)]

        self.layers = nn.ModuleList(*[layers])


    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False, checkpoint_activations=False):
        #assert output_all_encoded_layers == False
        assert checkpoint_activations == False
        x = transpose_x(hidden_states) # [N, W, C] --> [N, C, W]

        attention_scores = []
        feature_maps = []
        for layer in self.layers:
            x = layer.forward(x, attention_mask, output_all_encoded_layers)
            if output_all_encoded_layers:
                attention_scores.append(x['attention_score'])
                feature_maps.append(x['feature_map'])
                x = x['feature_map'] # to feed to next layer

        x = transpose_x(x) # [N, C, W] --> [N, W, C]
        result = {'encoded_layers': [x], 'cost': self.cost} # TODO(forresti): remove self.cost before releasing, unless we are doing a NAS release

        if output_all_encoded_layers:
            result['attention_scores'] = attention_scores
            result['feature_maps'] = feature_maps

        return result

class SqueezeBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class SqueezeBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = SqueezeBertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, SqueezeBertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class SqueezeBertModel(SqueezeBertPreTrainedModel):
    """
    .. _`SqueezeBert`:
        https://arxiv.org/abs/2006.11316

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = SqueezeBertEmbeddings(config)
        self.encoder = SqueezeBertEncoder(config)
        self.pooler = SqueezeBertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="squeezebert-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )