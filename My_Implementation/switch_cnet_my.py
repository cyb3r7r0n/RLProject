from torch import nn
from torch.autograd import Variable
import torch


class SwitchCNetMy(nn.Module):

    def __init__(self, opt):

        super(SwitchCNetMy, self).__init__()

        self.opt = opt
        self.comm_size = opt["game_comm_bits"]
        self.init_param_range = (-0.08, 0.08)

        self.agent_lookup = nn.Embedding(opt["game_nagents"], opt["model_rnn_size"])
        self.state_lookup = nn.Embedding(2, opt["model_rnn_size"])

        self.prev_message_lookup = None

        self.prev_action_lookup = nn.Embedding(opt["game_action_space"]+1, opt["model_rnn_size"])
        self.prev_message_lookup = nn.Embedding(opt["game_comm_bits"]+1, opt["model_rnn_size"])

        self.messages_mlp = nn.Sequential()
        # self.messages_mlp.add_module('batchnorm1', nn.BatchNorm1d(self.comm_size))
        self.messages_mlp.add_module('linear1', nn.Linear(self.comm_size, opt["model_rnn_size"]))
        self.messages_mlp.add_module('relu1', nn.ReLU(inplace=True))

        self.rnn = nn.GRU(input_size=opt["model_rnn_size"], hidden_size=opt["model_rnn_size"], num_layers=opt["model_rnn_layers"], dropout=opt["model_rnn_dropout_rate"], batch_first=True)

        self.outputs = nn.Sequential()

        dropout_rate = opt["model_rnn_dropout_rate"]
        if dropout_rate > 0:
            self.outputs.add_module('dropout1', nn.Dropout(dropout_rate))

        self.outputs.add_module('linear1', nn.Linear(opt["model_rnn_size"], opt["model_rnn_size"]))
        # self.outputs.add_module('batchnorm1', nn.BatchNorm1d(opt["model_rnn_size"]))

        self.outputs.add_module('relu1', nn.ReLU(inplace=True))
        self.outputs.add_module('linear2', nn.Linear(opt["model_rnn_size"], opt["game_action_space_total"]))

    def get_params(self):
        return list(self.parameters())

    def reset_parameters(self):
        self.messages_mlp.linear1.reset_parameters()
        self.rnn.reset_parameters()
        self.agent_lookup.reset_parameters()
        self.state_lookup.reset_parameters()
        self.prev_action_lookup.reset_parameters()
        self.prev_message_lookup.reset_parameters()
        self.outputs.linear1.reset_parameters()
        self.outputs.linear2.reset_parameters()
        for p in self.rnn.parameters():
            p.data.uniform_(*self.init_param_range)

    def forward(self, s_t, messages, hidden, prev_action, agent_index):

        s_t = Variable(s_t)
        hidden = Variable(hidden)

        prev_action, prev_message = prev_action
        prev_action = Variable(torch.tensor(prev_action, dtype=torch.long))
        prev_message = Variable(torch.tensor(prev_message, dtype=torch.long))
        agent_index = Variable(torch.tensor(agent_index, dtype=torch.long))

        z_a = self.agent_lookup(agent_index)
        z_o = self.state_lookup(s_t)
        z_u = self.prev_action_lookup(prev_action)
        if prev_message is not None:
            z_u += self.prev_message_lookup(prev_message)

        # print(messages.view(-1, self.comm_size))
        z_m = self.messages_mlp(messages.view(-1, self.comm_size))

        z = z_a + z_o + z_u + z_m
        z = z.unsqueeze(1)

        # print("Shape of hidden before : ", hidden.shape)
        hidden = hidden.view(2,1,128)
        # print("Shape of hidden before : ", hidden.shape)

        rnn_out, h_out = self.rnn(z, hidden)

        # print(rnn_out.shape)
        outputs = self.outputs(rnn_out[:, -1, :].squeeze())

        return h_out, outputs
