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

from arch_search import MixedModule
from modeling_util import gelu, swish, ACT2FN, MatMulWrapper, transpose_x, is_tracing

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

# TODO(forresti): replace assert statements with the following to get rid of warnings in tracing mode.
def _assert(case, message):
    is_tracing_mode = (torch._C._get_tracing_state() is not None)
    if not is_tracing_mode:
        assert case, message

class YotaSuperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = 'conv1d'

    # TODO(forresti): do something like SuperNetwork, where we can update all children that are YotaModules.
    #  Specifically, add a switch_mode() function, which calls switch_mode on all child YotaModules

class YotaModule(nn.Module, ABC):
    def __init__(self):
        nn.Module.__init__(self)
        ABC.__init__(self)
        self.mode = 'conv1d'
        self.forward = self.forward_conv1d

    def switch_mode(self, new_mode):
        assert new_mode in ['conv1d', 'linear'], f"unsupported mode {new_mode}"
        self.mode = new_mode

        if new_mode == 'conv1d':
            self.forward = self.forward_conv1d
        elif new_mode == 'linear':
            self.forward = self.forward_linear

    def forward_conv1d(self, *args, **kwargs):
        return NotImplemented

    def linear(self, *args, **kwargs):
        return NotImplemented

try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction
    FusedLayerNorm_is_available = True
except:
    pass

class SqueezeBertLayerNorm(nn.LayerNorm, YotaModule):
    def __init__(self, hidden_size, eps=1e-12):
        YotaModule.__init__(self)
        nn.LayerNorm.__init__(self, normalized_shape=hidden_size, eps=eps) # instantiates self.{weight, bias, eps}
        self.shape = torch.Size((hidden_size,)) # for FusedLayerNormAffineFunction

    # no need to override YotaModule's self.switch_mode(), because nothing needs to be transposed.

    def apply_layernorm(self, x):
        if FusedLayerNorm_is_available and (not is_tracing()) and ('cuda' in x.device.type):
            return FusedLayerNormAffineFunction.apply(x, self.weight, self.bias, self.shape, self.eps)
        else:
            return nn.LayerNorm.forward(self, x)

    def forward_conv1d(self, x):
        x = x.permute(0, 2, 1)
        x = self.apply_layernorm(x)
        return x.permute(0, 2, 1)

    def forward_linear(self, x):
        return self.apply_layernorm(x)


class Conv1d_yota(nn.Conv1d, YotaModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros'):
        YotaModule.__init__(self)
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        # note that nn.Conv1d automatically sets self.in_channels=in_channels, self.stride=stride, etc.
        # so, we retain access to the input arguments

    def switch_mode(self, new_mode):
        """
        self.mode='conv1d':
            self.weight.shape = [cout, cin, ksize]
        self.mode='linear':
            self.weight.shape = [cout, cin]
        """
        if new_mode == self.mode:
            return # no mode-change needed

        elif new_mode == 'conv1d':
            assert self.mode == 'linear' # we are switching from linear to conv1d
            self.weight.data = self.weight.data.unsqueeze(-1) # [cout, cin] --> [cout, cin, ksize=1]

        elif new_mode == 'linear':
            required_settings = {'kernel_size': (1,), # note that when you input kernel_size=1, it gets saved as a length-1 tuple.
                                 'stride': (1,),
                                 'padding': (0,),
                                 'dilation': (1,),
                                 'groups': 1}
            for setting_name, required_value in required_settings.items():
                curr_value = getattr(self, setting_name)
                if curr_value != required_value:
                    raise ValueError(f"Unable to switch to mode='linear', because self.{setting_name}={curr_value}, "
                                     f"and linear mode requires self.{setting_name}={required_value}")

            self.weight.data = self.weight.data.squeeze(-1)  # [cout, cin, ksize=1] --> [cout, cin]

        else:
            raise ValueError(f'requested unknown new_mode: {new_mode}')

        YotaModule.switch_mode(self, new_mode) # update self.mode

    def forward_conv1d(self, x):
        return nn.Conv1d.forward(self, x) # TODO(forresti): should this have .forward, or not? Does it matter?

    def forward_linear(self, x):
        num_weight_dims = len(list(self.weight.shape))
        assert num_weight_dims == 2, f"forward_linear() expects self.weight to have 2 dimensions, but got" \
                                     f" {num_weight_dims} dimensions."

        # note that, if bias was set to False in __init__(), then self.bias=None, which is compatible with the following
        return F.linear(input=x, weight=self.weight, bias=self.bias)


class CDL(nn.Module):
    # TODO(forresti): should mid-level classes like this also be YotaSuperModules, so they can call switch_mode() on child YotaModules?
    '''
    CDL: Conv, Dropout, LayerNorm
    '''
    def __init__(self, cin, cout, groups, dropout_prob):
        super().__init__()

        self.conv1d = Conv1d_yota(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.layernorm = SqueezeBertLayerNorm(cout)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        x = self.conv1d(hidden_states)
        x = self.dropout(x)
        x = x + input_tensor
        x = self.layernorm(x)
        return x


class CA(nn.Module):
    # TODO(forresti): should mid-level classes like this also be YotaSuperModules, so they can call switch_mode() on child YotaModules?
    '''
    CA: Conv, Activation
    '''
    def __init__(self, cin, cout, groups, act):
        super().__init__()
        self.conv1d = Conv1d_yota(in_channels=cin, out_channels=cout, kernel_size=1, groups=groups)
        self.act = ACT2FN[act]

    def forward(self, x):
        output = self.conv1d(x)
        return self.act(output)

class SqueezeBertSelfAttention(YotaModule):
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

        self.query = Conv1d_yota(in_channels=cin, out_channels=cin, kernel_size=1, groups=q_groups)
        self.key   = Conv1d_yota(in_channels=cin, out_channels=cin, kernel_size=1, groups=k_groups)
        self.value = Conv1d_yota(in_channels=cin, out_channels=cin, kernel_size=1, groups=v_groups)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1) # this is unchanged from original code, because it's applied to the output of (Q K^T V), which is in the same format here as in the old code

        self.matmul_qk = MatMulWrapper()
        self.matmul_qkv = MatMulWrapper()

    # Note: We don't have a switch_mode() here that updates the mode of the child YotaModules.
    # That is because the BertModule is supposed to update the mode of all child YotaModules, including this and its children.

    def transpose_for_scores_conv1d(self, x):
        '''
        input: [N, C, W]
        output: [N, C1, W, C2]
            where C1 is the head index, and C2 is one head's contents
        '''
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1]) # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2) # [N, C1, C2, W] --> [N, C1, W, C2]

    def transpose_for_scores_linear(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores_conv1d(self, x):
        '''
        input: [N, C, W]
        output: [N, C1, C2, W]
            where C1 is the head index, and C2 is one head's contents
        '''
        new_x_shape = (x.size()[0], self.num_attention_heads, self.attention_head_size, x.size()[-1]) # [N, C1, C2, W]
        x = x.view(*new_x_shape)
        # no `permute` needed
        return x

    def transpose_key_for_scores_linear(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 3, 1)

    def transpose_output_conv1d(self, x):
        '''
        input: [N, C1, W, C2]
        output: [N, C, W]
        '''
        x = x.permute(0, 1, 3, 2).contiguous() # [N, C1, C2, W]
        new_x_shape = (x.size()[0], self.all_head_size, x.size()[3]) # [N, C, W]
        x = x.view(*new_x_shape)
        return x

    def transpose_output_linear(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (self.all_head_size,) # [N, W, C]
        x = x.view(*new_x_shape)
        return x

    def forward_conv1d(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        '''
        expects hidden_states in NCW form.
        '''
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores_conv1d(mixed_query_layer)
        key_layer = self.transpose_key_for_scores_conv1d(mixed_key_layer)
        value_layer = self.transpose_for_scores_conv1d(mixed_value_layer)

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
        context_layer = self.transpose_output_conv1d(context_layer)
        if output_all_encoded_layers:
            return {'attention_scores': attention_scores, 'context_layer': context_layer} # note that TinyBERT also uses attention_scores for distillation
        else:
            return context_layer

    def forward_linear(self, hidden_states, attention_mask):
        # TODO(forresti): possibly support output_all_encoded_layers in forward_linear
        '''
        expects hidden_states in NWC form.
        '''
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores_linear(mixed_query_layer)
        key_layer = self.transpose_key_for_scores_linear(mixed_key_layer)
        value_layer = self.transpose_for_scores_linear(mixed_value_layer)

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
        context_layer = self.transpose_output_linear(context_layer)
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


class SqueezeBertEncoder(YotaModule):
    # TODO(forresti): make this a YotaSuperModule, so it can update the children's modes
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
        self.register_buffer('cost', torch.zeros(1)) # TODO(forresti): remove self.cost before releasing, unless we are doing a NAS release

        if hasattr(config, 'embedding_size'):
            assert config.embedding_size == hidden_size, "if you want embedding_size != intermediate hidden_size, " \
                                                         "please add a Conv1d_yota layer beofre the first BertModule " \
                                                         "that adjusts the number of channels."

        layers = [SqueezeBertModule(config, hidden_size=hidden_size, intermediate_size=intermediate_size,
                                          q_groups=q_groups, k_groups=k_groups, v_groups=v_groups,
                                          intermediate_groups=intermediate_groups, post_attention_groups=post_attention_groups,
                                          output_groups=output_groups)
                                          for _ in range(config.num_hidden_layers)]

        self.layers = nn.ModuleList(*[layers])

    def switch_mode(self, new_mode):
        # Update the modes of all child modules (at all levels of the model graph)
        for name, module in self.named_modules():
            if isinstance(module, YotaModule) and (module is not self): # TODO(forresti): do I need this 'module is not self' to prevent an infinite recursion here?
                module.switch_mode(new_mode)

        super().switch_mode(new_mode) # TODO(forresti): If I create a YodaSuperModule, I may need to say YotaModule instead of super() here.

    def forward_conv1d(self, hidden_states, attention_mask, output_all_encoded_layers=False, checkpoint_activations=False):
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

    def forward_linear(self, hidden_states, attention_mask, output_all_encoded_layers=False, checkpoint_activations=False):
        assert output_all_encoded_layers == False
        assert checkpoint_activations == False
        x = hidden_states

        # note the lack of transpose_x here, because the input, encoder, and output all use the [N, W, C] data layout.
        for layer in self.layers:
            x = layer.forward(x, attention_mask)

        # TODO(forresti): add code for checkpoint_activations in linear mode

        return {'encoded_layers': [x], 'cost': self.cost} # TODO(forresti): remove self.cost before releasing, unless we are doing a NAS release

