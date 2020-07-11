# coding:utf-8
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model.torch_crf import CRF
from layers.dynamic_rnn import DynamicLSTM
import time
from data_util import config


class Joint_model(nn.Module):
    def __init__(self, _, hidden_dim, batch_size, max_length, n_class, n_tag, embedding_matrix):
        super(Joint_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_class = n_class
        self.n_tag = n_tag
        self.LayerNorm = LayerNorm(self.hidden_dim, eps=1e-12)
        self.emb_drop = nn.Dropout(config.emb_dorpout)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = True
        self.biLSTM = DynamicLSTM(config.emb_dim, config.hidden_dim // 2, bidirectional=True, batch_first=True,
                                  dropout=config.lstm_dropout, num_layers=1)
        self.intent_fc = nn.Linear(self.hidden_dim, self.n_class)
        self.slot_fc = nn.Linear(self.hidden_dim, self.n_tag)
        self.I_S_Emb = Label_Attention(self.intent_fc, self.slot_fc)
        self.T_block1 = I_S_Block(self.intent_fc, self.slot_fc, self.hidden_dim)
        self.T_block2 = I_S_Block(self.intent_fc, self.slot_fc, self.hidden_dim)
        self.T_block3 = I_S_Block(self.intent_fc, self.slot_fc, self.hidden_dim)
        self.crflayer = CRF(self.n_tag)
        self.criterion = nn.CrossEntropyLoss()

    def forward_logit(self, x, mask):
        x, x_char = x
        x_len = torch.sum(x != 0, dim=-1)
        x_emb = self.emb_drop(self.embed(x))

        H, (_, _) = self.biLSTM(x_emb, x_len)
        H_I, H_S = self.I_S_Emb(H, H, mask)
        H_I, H_S = self.T_block1(H_I + H, H_S + H, mask)
        H_I_1, H_S_1 = self.I_S_Emb(H_I, H_S, mask)
        H_I, H_S = self.T_block2(H_I + H_I_1, H_S + H_S_1, mask)

        intent_input = F.max_pool1d((H_I + H).transpose(1, 2), H_I.size(1)).squeeze(2)
        logits_intent = self.intent_fc(intent_input)
        logits_slot = self.slot_fc(H_S + H)

        return logits_intent, logits_slot

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        slot_label = slot_label[:, 0:logits_slot.size(1)]
        logits_slot = logits_slot.transpose(1, 0)
        slot_label = slot_label.transpose(1, 0)
        mask = mask.transpose(1, 0)
        loss_intent = self.criterion(logits_intent, intent_label)
        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super(Intermediate, self).__init__()
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states_in):
        hidden_states = self.dense_in(hidden_states_in)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + hidden_states_in)
        return hidden_states


class Intermediate_I_S(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super(Intermediate_I_S, self).__init__()
        self.dense_in = nn.Linear(hidden_size * 6, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm_I = LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm_S = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states_I, hidden_states_S):
        hidden_states_in = torch.cat([hidden_states_I, hidden_states_S], dim=2)
        batch_size, max_length, hidden_size = hidden_states_in.size()
        h_pad = torch.zeros(batch_size, 1, hidden_size)
        if config.use_gpu and torch.cuda.is_available():
            h_pad = h_pad.cuda()
        h_left = torch.cat([h_pad, hidden_states_in[:, :max_length - 1, :]], dim=1)
        h_right = torch.cat([hidden_states_in[:, 1:, :], h_pad], dim=1)
        hidden_states_in = torch.cat([hidden_states_in, h_left, h_right], dim=2)

        hidden_states = self.dense_in(hidden_states_in)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states_I_NEW = self.LayerNorm_I(hidden_states + hidden_states_I)
        hidden_states_S_NEW = self.LayerNorm_S(hidden_states + hidden_states_S)
        return hidden_states_I_NEW, hidden_states_S_NEW


class I_S_Block(nn.Module):
    def __init__(self, intent_emb, slot_emb, hidden_size):
        super(I_S_Block, self).__init__()
        self.I_S_Attention = I_S_SelfAttention(hidden_size, 2 * hidden_size, hidden_size)
        self.I_Out = SelfOutput(hidden_size, config.attention_dropout)
        self.S_Out = SelfOutput(hidden_size, config.attention_dropout)
        self.I_S_Feed_forward = Intermediate_I_S(hidden_size, hidden_size)

    def forward(self, H_intent_input, H_slot_input, mask):
        H_slot, H_intent = self.I_S_Attention(H_intent_input, H_slot_input, mask)
        H_slot = self.S_Out(H_slot, H_slot_input)
        H_intent = self.I_Out(H_intent, H_intent_input)
        H_intent, H_slot = self.I_S_Feed_forward(H_intent, H_slot)

        return H_intent, H_slot


class Label_Attention(nn.Module):
    def __init__(self, intent_emb, slot_emb):
        super(Label_Attention, self).__init__()

        self.W_intent_emb = intent_emb.weight
        self.W_slot_emb = slot_emb.weight

    def forward(self, input_intent, input_slot, mask):
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())
        intent_probs = nn.Softmax(dim=-1)(intent_score)
        slot_probs = nn.Softmax(dim=-1)(slot_score)
        intent_res = torch.matmul(intent_probs, self.W_intent_emb)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)

        return intent_res, slot_res


class I_S_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(I_S_SelfAttention, self).__init__()

        self.num_attention_heads = 8
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.out_size = out_size
        self.query = nn.Linear(input_size, self.all_head_size)
        self.query_slot = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.key_slot = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.out_size)
        self.value_slot = nn.Linear(input_size, self.out_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def transpose_for_scores(self, x):
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, intent, slot, mask):
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        mixed_query_layer = self.query(intent)
        mixed_key_layer = self.key(slot)
        mixed_value_layer = self.value(slot)

        mixed_query_layer_slot = self.query_slot(slot)
        mixed_key_layer_slot = self.key_slot(intent)
        mixed_value_layer_slot = self.value_slot(intent)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer_slot = self.transpose_for_scores(mixed_query_layer_slot)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        key_layer_slot = self.transpose_for_scores(mixed_key_layer_slot)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        value_layer_slot = self.transpose_for_scores(mixed_value_layer_slot)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # attention_scores_slot = torch.matmul(query_slot, key_slot.transpose(1,0))
        attention_scores_slot = torch.matmul(query_layer_slot, key_layer_slot.transpose(-1, -2))
        attention_scores_slot = attention_scores_slot / math.sqrt(self.attention_head_size)
        attention_scores_intent = attention_scores + attention_mask

        attention_scores_slot = attention_scores_slot + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)
        attention_probs_intent = nn.Softmax(dim=-1)(attention_scores_intent)

        attention_probs_slot = self.dropout(attention_probs_slot)
        attention_probs_intent = self.dropout(attention_probs_intent)

        context_layer_slot = torch.matmul(attention_probs_slot, value_layer_slot)
        context_layer_intent = torch.matmul(attention_probs_intent, value_layer)

        context_layer = context_layer_slot.permute(0, 2, 1, 3).contiguous()
        context_layer_intent = context_layer_intent.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        new_context_layer_shape_intent = context_layer_intent.size()[:-2] + (self.out_size,)

        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_intent = context_layer_intent.view(*new_context_layer_shape_intent)
        return context_layer, context_layer_intent
