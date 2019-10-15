import torch
import torch.nn as nn


# Model
class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers, drop_prob):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=layers, batch_first=True,
            dropout=drop_prob)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=drop_prob)
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
        x, _status = self.rnn(x)
        x = self.dropout1(x[:, -1])
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    # Initialization parameter
    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)


# Put data in tensor
def get_tensor(device, data_x, data_y, start_idx, end_idx):
    x_tensor = torch.FloatTensor(data_x[start_idx:end_idx, :, :]).to(device)
    y_tensor = torch.FloatTensor(data_y[start_idx:end_idx])
    y_tensor.unsqueeze_(-1)
    y_tensor = y_tensor.to(device)
    return x_tensor, y_tensor


# Use GRU or CPU
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(777)
        print('Using CUDA')
    else:
        device = 'cpu'
        print('Using CPU')
    torch.manual_seed(777)
    return device


# Normalize in tensor
def normalize_tensor(tensor, minmax):
    min_val = minmax[0]
    max_val = minmax[1]
    arr = tensor.cpu().data.numpy()*(max_val-min_val) + min_val
    return arr


# Train
def train(device, start, rate, datax, datay):
    train_x_tensor, train_y_tensor = get_tensor(
        device, datax, datay, start, start + rate)
    return train_x_tensor, train_y_tensor


# Validation
def valid(device, start, rate, datax, datay):
    valid_x_tensor, valid_y_tensor = get_tensor(
        device, datax, datay, start, start + rate)
    return valid_x_tensor, valid_y_tensor