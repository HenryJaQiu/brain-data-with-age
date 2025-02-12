import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence


# Model
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, drop_prob,
                 model_name, bidirection=False):
        super(RNNClassifier, self).__init__()
        if model_name == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers,
                              batch_first=True, bidirectional=bidirection)
        elif model_name == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                               batch_first=True, bidirectional=bidirection)
        elif model_name == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=layers,
                              batch_first=True, bidirectional=bidirection)
        else:
            raise ValueError("Check the model name!(RNN/ LSTM/ GRU)")

        self.dropout = nn.Dropout(p=drop_prob)

        if bidirection:
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

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
        result, _status = self.rnn(x)
        out, lens_unpacked = pad_packed_sequence(result, batch_first=True)
        lens_unpacked = (lens_unpacked - 1).unsqueeze(1).unsqueeze(2)
        index = lens_unpacked.expand(lens_unpacked.size(0),
                                     lens_unpacked.size(1),
                                     out.size(2))
        x = torch.gather(out, 1, index).squeeze()
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.fc2(self.dropout(x))
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