import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
import models.dornn_simple as ftru
import models.layers as mrnns
import models.roseyu_trnn as hotrnn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MRNNFixD(nn.Module):
    """mRNN with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.mrnn_cell = mrnns.MRNNFixDCell(input_size, hidden_size,
                                             output_size, k)
    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_ws(d_value[0, hidden].
                                                         view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        self.d_matrix = 0.5 * F.sigmoid(self.b_d)
        weights_d = self.get_wd(self.d_matrix)
        for times in range(time_steps):
            outputs, hidden_state = self.mrnn_cell(inputs[times, :], weights_d,
                                                   hidden_state)
        return outputs, hidden_state


class MRNN(nn.Module):
    """mRNN with dynamic d for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNN, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.mrnn_cell = mrnns.MRNNCell(input_size, hidden_size,
                                         output_size, k)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.Tensor(time_steps, batch_size, self.output_size)
        for times in range(time_steps):
            outputs[times, :], hidden_state = self.mrnn_cell(inputs[times, :],
                                                             hidden_state)
        return outputs, hidden_state

class MLSTMFixD(nn.Module):
    """mLSTM with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTMFixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.d_values = Parameter(torch.Tensor(torch.zeros(1, hidden_size)),
                                  requires_grad=True)
        self.output_size = output_size
        self.mlstm_cell = mrnns.MLSTMFixDCell(self.input_size,
                                               self.hidden_size,
                                               self.output_size,
                                               self.k)
        self.sigmoid = nn.Sigmoid()

    def get_w(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_w(d_value[0, hidden].
                                                        view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, hidden_states=None):
        if hidden_states is None:
            hidden = None
            h_c = None
        else:
            hidden = hidden_states[0]
            h_c = hidden_states[1]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        self.d_values_sigmoid = 0.5 * F.sigmoid(self.d_values)
        weights_d = self.get_wd(self.d_values_sigmoid)
        for times in range(time_steps):
            outputs[times, :], hidden, h_c = self.mlstm_cell(inputs[times, :],
                                                             hidden,
                                                             h_c,
                                                             weights_d)
        return outputs, (hidden, h_c)


class MLSTM(nn.Module):
    """mLSTM with dynamic d for time series prediction"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size
        self.mlstm_cell = mrnns.MLSTMCell(self.input_size, self.hidden_size,
                                           self.k, self.output_size)

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            hidden = None
            h_c = None
            d_values = None
        else:
            hidden = hidden_state[0]
            h_c = hidden_state[1]
            d_values = hidden_state[2]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        for times in range(time_steps):
            outputs[times, :], hidden, h_c, d_values = \
                self.mlstm_cell(inputs[times, :], hidden, h_c, d_values)
        return outputs, (hidden, h_c, d_values)


# Model
class TensorClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, drop_prob,
                 model_name, bidirection=False):
        super(TensorClassifier, self).__init__()
        if model_name == 'MRNN':
            self.rnn = MRNN(input_dim, hidden_dim, output_size=1, k=100)
        elif model_name == 'FTRU':
            self.rnn = ftru.DORNN(input_dim, hidden_dim, output_size=1, prefix='HN1lCDx1Dh1Dp3')
        else:
            raise ValueError("Check the model name!(MRNN/ LSTM/ GRU)")

        self.dropout = nn.Dropout(p=drop_prob)

        if bidirection:
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        else:
            self.fc1 = nn.Linear(100, hidden_dim, bias=True)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # Initialize RNN Module
        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        # Initialize Linear Module
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.normal_(self.fc2.bias.data)
        # If you want to guess age: output_dim=1
        # If you want to categorize age: output_dim = number of categories

    def forward(self, x):
        '''
        result, _status = self.rnn(x)
        out, lens_unpacked = pad_packed_sequence(result, batch_first=True)
        lens_unpacked = (lens_unpacked - 1).unsqueeze(1).unsqueeze(2)
        # index = lens_unpacked.expand(lens_unpacked.size(0), lens_unpacked.size(1), out.size(2)).cuda()
        index = lens_unpacked.expand(lens_unpacked.size(0), lens_unpacked.size(1), out.size(2))  # CPU only
        x = torch.gather(out, 1, index).squeeze()
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(self.dropout(x))
        '''

        x, _status = self.rnn(x)
        x = torch.squeeze(x)
        print(x.size())
        x1 = self.fc1(x)
        x2 = self.bn1(x1)
        x3 = self.relu(x2)
        x = self.fc2(self.dropout(x3))
        #for layer in self.hidden:
        #    x = layer(x)

        return x


    # Initialization parameter
    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN, nn.Linear]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                    elif 'weight' in name:
                        torch.nn.init.xavier_uniform_(param.data)
