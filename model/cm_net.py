# coding:utf-8
import torch
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from model.torch_crf import CRF
from model.cell import SLSTMCell
# from .LSTM_layernormalization import LSTM
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
        self.emb_drop = nn.Dropout(0.5)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), padding_idx=0)
        self.embed.weight.requires_grad = False
        self.emb_char = WordCharCNNEmbedding()
        self.cm_hidden_size = 400
        self.local_cal = SLSTMCell(self.cm_hidden_size, self.cm_hidden_size)
        self.biLSTM = DynamicLSTM(400, 400//2, bidirectional=True, batch_first=True,
                                  dropout=0.5, num_layers=1)
        self.intent_memory = nn.Linear(self.cm_hidden_size, self.n_class)
        self.slot_memory = nn.Linear(self.cm_hidden_size, self.n_tag)

        self.intent_fc = nn.Linear(self.cm_hidden_size * 2, self.n_class)
        self.slot_fc = nn.Linear(self.cm_hidden_size * 2, self.n_tag)
        self.I_S_Emb = Label_Attention(self.intent_memory, self.slot_memory)

        self.crflayer = CRF(self.n_tag)
        self.criterion = nn.CrossEntropyLoss()

    def forward_logit(self, x, mask):
        x, x_char = x
        x_len = torch.sum(x != 0, dim=-1)
        x_emb = self.emb_drop(self.embed(x))
        x_char_emb = self.emb_char(x_char)
        batch_size, max_length, emb_size = x_emb.size()
        x_emb = torch.cat([x_emb, x_char_emb], dim=2)
        C_t = torch.zeros(batch_size, max_length, self.cm_hidden_size).cuda()

        # layer 1
        h = x_emb
        I, S = self.I_S_Emb(h, mask)
        h_int = torch.cat([h, S],dim=2)
        h_slot = torch.cat([h, I], dim=2)
        H_I, H_S = self.I_S_Emb.forward_cat(h_int, h_slot ,mask)
        Out_state, C_t = self.local_cal(h, x_emb, H_I, H_I, C_t)
        H_NEW, (_, _) = self.biLSTM(Out_state, x_len)

        # layer 2
        h = H_NEW
        I, S = self.I_S_Emb(h, mask)
        h_int = torch.cat([h, S],dim=2)
        h_slot = torch.cat([h, I], dim=2)
        H_I, H_S = self.I_S_Emb.forward_cat(h_int, h_slot, mask)
        Out_state, C_t = self.local_cal(h, x_emb, H_I, H_I, C_t)
        H_NEW, (_, _) = self.biLSTM(Out_state, x_len)

        # layer 3
        h = H_NEW
        I, S = self.I_S_Emb(h, mask)
        h_int = torch.cat([h, S],dim=2)
        h_slot = torch.cat([h, I], dim=2)
        H_I, H_S = self.I_S_Emb.forward_cat(h_int, h_slot, mask)
        Out_state, C_t = self.local_cal(h, x_emb, H_I, H_I, C_t)
        H_NEW, (_, _)= self.biLSTM(Out_state, x_len)


        out_intent = torch.cat([H_NEW, H_I], dim=2)
        out_slot = torch.cat([H_NEW,H_S], dim=2)
        intent_input = F.avg_pool1d((out_intent).transpose(1, 2), out_intent.size(1)).squeeze(2)
        logits_intent = self.intent_fc(intent_input)
        logits_slot = self.slot_fc(out_slot)

        return logits_intent, logits_slot

    def loss1(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        mask = mask[:, 0:logits_slot.size(1)]
        slot_label = slot_label[:, 0:logits_slot.size(1)]
        logits_slot = logits_slot.transpose(1, 0)
        slot_label = slot_label.transpose(1, 0)
        mask = mask.transpose(1, 0)
        #         loss_intent1 = self.criterion(logits_list[0][0], intent_label)
        #         loss_intent2 = self.criterion(logits_list[1][0], intent_label)
        loss_intent = self.criterion(logits_intent, intent_label)
        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]

        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        t0 = time.time()
        x_len = torch.sum(mask != 0, dim=1)
        mask = mask[:, 0:logits_slot.size(1)]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_intent = torch.max(logits_intent, 1)[1]
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        # print("crf解码耗时::", time.time() - t0)
        return pred_intent, pred_slot


class WordCharCNNEmbedding(nn.Module):
    """Combination between character and word embedding as the
    features for the tagger. The character embedding is built
    upon CNN and pooling layer.
    """

    def __init__(self,
                 char_num_embedding=29,
                 char_embedding_dim=100,
                 char_padding_idx=0,
                 dropout=0.5,
                 kernel_size=3,
                 out_channels=100,
                 pretrained_word_embedding=None):
        super(WordCharCNNEmbedding, self).__init__()
        self.char_embedding = nn.Embedding(
            char_num_embedding, char_embedding_dim, padding_idx=char_padding_idx)
        self._init_char_embedding(char_padding_idx)
        self.conv_embedding = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv1d(
                in_channels=100,
                out_channels=100,
                kernel_size=3))

        if isinstance(pretrained_word_embedding, torch.Tensor):
            self.word_embedding.weight.data.copy_(pretrained_word_embedding)
            # Freeze the embedding layer when using pretrained word
            # embedding
            self.word_embedding.weight.requires_grad = False

    def _init_char_embedding(self, padding_idx):
        """Initialize the weight of character embedding with xavier
        and reinitalize the padding vectors to zero
        """

        nn.init.xavier_normal_(self.char_embedding.weight)
        # Reinitialize vectors at padding_idx to have 0 value
        self.char_embedding.weight.data[padding_idx].uniform_(0, 0)

    def forward(self, X):
        word_size = X.size(1)
        char_embeddings = []
        # We have X in dimension of [batch, words, chars]. To use
        # batch calculation we need to loop over all words and
        # calculate the embedding
        char_embeddings = self.char_embedding(X)
        char_embeddings = char_embeddings.view(-1, 25, 100)
        char_embeddings = self.conv_embedding(char_embeddings.permute(0, 2, 1))
        char_embeddings = F.max_pool1d(char_embeddings, 23).squeeze(-1).view(-1, 32, 100)
        final_char_embedding = char_embeddings
        return final_char_embedding





class Label_Attention(nn.Module):
    def __init__(self, intent_emb, slot_emb):
        super(Label_Attention, self).__init__()

        self.W_intent_emb = intent_emb.weight
        self.W_slot_emb = slot_emb.weight
        self.W_1 = nn.Linear(400, 400, bias=None)
        self.W_2 = nn.Linear(800, 400, bias=None)

        # self.W_intent = nn.Linear(400, 800, bias=None)
        # self.W_slot = nn.Linear(400, 800, bias=None)
        # self.attention_head_size = 128

    def forward(self, H, mask):
        # x_len = torch.sum(mask != 0, dim=-1)
        #         input_intent= input_intent * mask.float().unsqueeze(2)
        #         input_intent = torch.sum(input_intent,dim=1,keepdim=True)
        #         input_intent = input_intent /x_len.unsqueeze(1).unsqueeze(2).float()
        #         #input_intent = F.avg_pool1d(input_intent.transpose(1, 2), input_intent.size(1)).transpose(1, 2)
        # input_intent1 = F.max_pool1d(input_intent.transpose(1, 2), input_intent.size(1)).transpose(1, 2)
        input_intent = self.W_1(H)
        input_slot = self.W_1(H)
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())

        # intent_score = intent_score / math.sqrt(self.attention_head_size)
        # slot_score = slot_score / math.sqrt(self.attention_head_size)

        intent_probs = nn.Softmax(dim=-1)(intent_score)
        slot_probs = nn.Softmax(dim=-1)(slot_score)
        #         i = intent_probs.squeeze(0).cpu().detach().numpy()
        #         s = slot_probs.squeeze(0).cpu().detach().numpy()
        # s = slot_probs.cpu().detach().numpy()
        intent_res = torch.matmul(intent_probs, self.W_intent_emb)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)

        return intent_res, slot_res

    def forward_cat(self, intent, slot, mask):

        input_intent = self.W_2(intent)
        input_slot = self.W_2(slot)
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t())
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t())

        # intent_score = intent_score / math.sqrt(self.attention_head_size)
        # slot_score = slot_score / math.sqrt(self.attention_head_size)

        intent_probs = nn.Softmax(dim=-1)(intent_score)
        slot_probs = nn.Softmax(dim=-1)(slot_score)
        #         i = intent_probs.squeeze(0).cpu().detach().numpy()
        #         s = slot_probs.squeeze(0).cpu().detach().numpy()
        # s = slot_probs.cpu().detach().numpy()
        intent_res = torch.matmul(intent_probs, self.W_intent_emb)
        slot_res = torch.matmul(slot_probs, self.W_slot_emb)
        return intent_res, slot_res

class I_S_SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(I_S_SelfAttention, self).__init__()

        self.num_attention_heads = 8  # 暂为1
        self.attention_head_size = int(hidden_size / self.num_attention_heads)

        # all_head_size = hidden_size
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
        # tuple （1，2）+（3，4）=（1，2，3，4）
        last_dim = int(x.size()[-1] / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, last_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, intent, slot, mask):
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - extended_attention_mask) * -10000.0

        """
        Args:
            hidden_states: [batch_size, sequence_length, hidden_size]
            attention_mask: [batch_size, 1, 1, sequence_length]
        Returns:
            Tensor: [batch_size, sequence_length, hidden_size]
        """
        # [batch_size, sequence_length, hidden_size] -> [batch_size, sequence_length, all_head_size]
        mixed_query_layer = self.query(intent)
        mixed_key_layer = self.key(slot)
        mixed_value_layer = self.value(slot)

        mixed_query_layer_slot = self.query_slot(slot)
        mixed_key_layer_slot = self.key_slot(intent)
        mixed_value_layer_slot = self.value_slot(intent)
        # [batch_size, sequence_length, all_head_size] ->
        # [batch_size, sequence_length, num_attention_heads, attention_head_size]
        # [batch_size, sequence_length, num_attention_heads, attention_head_size] ->
        # [batch_size, num_attention_heads, sequence_length, attention_head_size]
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

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_mask:  [batch_size, 1, 1, sequence_length]
        # attention_scores:[batch_size, num_attention_heads, sequence_length, sequence_length]
        attention_scores_intent = attention_scores + attention_mask
        # attention_scores_intent = torch.sum(attention_scores_intent, dim=2, keepdim=True)

        attention_scores_slot = attention_scores_slot + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)
        attention_probs_intent = nn.Softmax(dim=-1)(attention_scores_intent)
        # attention_probs_slot = nn.Softmax(dim=-1)(attention_scores_slot)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_slot = self.dropout(attention_probs_slot)
        attention_probs_intent = self.dropout(attention_probs_intent)
        # attention_probs_slot = self.dropout(attention_probs_slot)
        # attention_probs: [batch_size, num_attention_heads, sequence_length, sequence_length]
        # value_layer    : [batch_size, num_attention_heads, sequence_length, attention_head_size]
        # context_layer  : [batch_size, num_attention_heads, sequence_length, attention_head_size]
        context_layer_slot = torch.matmul(attention_probs_slot, value_layer_slot)
        context_layer_intent = torch.matmul(attention_probs_intent, value_layer)
        # context_layer_slot = torch.matmul(attention_probs_slot, value_slot)
        # context_layer  : [batch_size, sequence_length, num_attention_heads, attention_head_size]
        context_layer = context_layer_slot.permute(0, 2, 1, 3).contiguous()
        context_layer_intent = context_layer_intent.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.out_size,)
        new_context_layer_shape_intent = context_layer_intent.size()[:-2] + (self.out_size,)
        # context_layer  : [batch_size, sequence_length, out_size]
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer_intent = context_layer_intent.view(*new_context_layer_shape_intent)
        return context_layer, context_layer_intent

