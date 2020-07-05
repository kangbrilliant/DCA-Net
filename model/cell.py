import torch
import torch.nn as nn
import torch.nn.functional as F


class HiddenGate(nn.Module):

    def __init__(self, hidden_size, input_size, bias, nonlinearity="sigmoid"):
        super(HiddenGate, self).__init__()
        self.linear = nn.Linear(
            hidden_size*6, hidden_size, bias=bias)
        self.nonlinearity = F.sigmoid if nonlinearity == "sigmoid" else F.tanh

    def forward(self, h, x, h_slot, h_intent):
        return self.nonlinearity(self.linear(torch.cat([h, x, h_slot, h_intent ],dim=-1)))


# class SentenceStateGate(nn.Module):
#
#     def __init__(self, hidden_size, input_size, bias):
#         super(SentenceStateGate, self).__init__()
#         self.linear = nn.Linear(
#             hidden_size + hidden_size, hidden_size, bias=bias)
#
#     def forward(self, prev_g, h):
#         """ h is either h_av or h_i for different i"""
#         return F.sigmoid(self.linear(torch.cat([prev_g, h])))


class SLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(SLSTMCell, self).__init__()
        self.input_size = input_size
        hidden_size = hidden_size
        self.hidden_size = hidden_size
        self.bias = bias

        # hidden state gates
        # 5个门
        self.i_t_op = HiddenGate(hidden_size, input_size, bias)
        self.o_t_op = HiddenGate(hidden_size, input_size, bias)
        self.f_t_op = HiddenGate(hidden_size, input_size, bias)
        self.l_t_op = HiddenGate(hidden_size, input_size, bias)
        self.r_t_op = HiddenGate(hidden_size, input_size, bias)


        self.u_i_op = HiddenGate(hidden_size, input_size,bias, nonlinearity="tanh")

        # sentence state gates
        #no g !!
        # self.g_f_g_op = SentenceStateGate(hidden_size, input_size, bias)
        # self.g_f_i_op = SentenceStateGate(hidden_size, input_size, bias)
        # self.g_o_op = SentenceStateGate(hidden_size, input_size, bias)

    def reset_params(self):
        pass

    def get_Xis(self, prev_h_states):
        """Apply proper index selection mask to get xis"""
        # How do you handle it getting shorter eh??
        pass
    def get_l_m_r(self,hidden_states ):

        batch_size, max_length, hidden_size = hidden_states.size()
        h_pad = torch.zeros(batch_size, 1, hidden_size).cuda()
        h_left = torch.cat([h_pad, hidden_states[:, :max_length - 1, :]], dim=1)
        h_right = torch.cat([hidden_states[:, 1:, :], h_pad], dim=1)
        cat_hidden_states = torch.cat([hidden_states, h_left, h_right], dim=2)

        return cat_hidden_states, h_left, hidden_states, h_right

    def forward(self, h_t, x, h_slot, h_intent, c_t):


        h_l_m_r,_,_,_ = self.get_l_m_r(h_t)

        i_t = self.i_t_op(h_l_m_r, x, h_slot, h_intent)
        o_t = self.o_t_op(h_l_m_r, x, h_slot, h_intent)
        f_t = self.f_t_op(h_l_m_r, x, h_slot, h_intent)
        l_t = self.l_t_op(h_l_m_r, x, h_slot, h_intent)
        r_t = self.r_t_op(h_l_m_r, x, h_slot, h_intent)

        u_t = self.u_i_op(h_l_m_r, x, h_slot, h_intent,)

        # Now Get Softmaxed Versions

        i_t, f_t, l_t, r_t = self.softmaxed_gates(
            [i_t, f_t, l_t, r_t])

        # what happens to the the last cell here?????? which has no i+1?
        # what happens when first one has no i-1??

        _,c_t_left, c_t_mid, c_t_right = self.get_l_m_r(c_t)

        c_t = f_t * c_t_mid + l_t * c_t_left + r_t * c_t_right + \
              i_t * u_t

        h_t = o_t * F.tanh(c_t)

        return h_t, c_t

    def softmaxed_gates(self, gates_list):
        softmaxed = F.softmax(torch.stack(gates_list), dim=0)
        return torch.unbind(softmaxed)
