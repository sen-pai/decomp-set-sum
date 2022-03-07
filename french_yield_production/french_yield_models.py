import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class YieldModel(nn.Module):
    def __init__(self, image_channels=6, h_dim=1024, z_dim=32):
        super(YieldModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(h_dim, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        pred = self.encoder(x)
        return torch.sum(pred)




class YieldRNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        n_layers=1,
        hidden_size=100,
        encoder_dropout=0.0,
        bidirectional=False,
    ):
        super(YieldRNNModel, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_size
        self.layers = n_layers
        self.dropout = encoder_dropout
        self.bi = bidirectional

        self.encoder_rnn = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=True,
        )

        self.reconstruct_linear = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, inputs, input_lengths):
        packed_input = pack_padded_sequence(
            inputs, input_lengths, enforce_sorted=False, batch_first=True
        )

        self.batch_size = inputs.size()[0]
        # print(self.batch_size)
        # max_len = int(torch.max(input_lengths).item())

        encoder_outputs, (h_n, c_n) = self.encoder_rnn(packed_input)
        encoder_outputs, _ = pad_packed_sequence(
            encoder_outputs, batch_first=True, total_length=6
        )

        # print(h_n.squeeze(0).shape)

        reconstructed = self.reconstruct_linear(h_n.squeeze(0))
        # print(reconstructed.shape)
        # encoder_outputs -> [batch size, max seq lenght, hidden size]
        # h_n -> [1, batch size, hidden size]
        # c_n -> [1, batch size, hidden size]

        # if self.bi:
        #     h_n = h_n.view(1, self.batch_size, self.hidden_size*2)
        #     c_n = c_n.view(1, self.batch_size, self.hidden_size*2)
            
        return torch.sum(reconstructed)

    def device(self) -> torch.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device




class Yasasd(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        n_layers=1,
        hidden_size=100,
        dropout=0.0,
        bidirectional=False,
        use_r_linear=True,  # adds a MLP after RNN predictions
    ):
        super().__init__(
            input_dim,
            output_dim,
            n_layers,
            hidden_size,
            dropout,
            bidirectional,
            use_r_linear,
        )

        self.reconstruct_rnn = nn.LSTM(
            self.input_size,
            self.output_dim if not self.use_r_linear else self.hidden_size,
            self.layers,
            dropout=self.dropout,
            bidirectional=self.bi,
            batch_first=False,
        )
        fin_h_size = self.hidden_size * 2 if self.bi else self.hidden_size

        fin_mid_size = int(fin_h_size / 2)

        self.reconstruct_linear = nn.Sequential(
            nn.Linear(fin_h_size, fin_mid_size),
            nn.ReLU(),
            nn.Linear(fin_mid_size, output_dim)
        )

    def single_step_deocde(self, prev_decode_output, h_i, c_i):
        # prev_decoder_output -> [1, batch size, 1]
        # encoder_outputs -> [batch size, max seq len, hidden size]
        # h_i -> [1, batch size, hidden size]
        output, (h_n, c_n) = self.reconstruct_rnn(prev_decode_output, (h_i, c_i))

        if self.use_r_linear:
            output = self.reconstruct_linear(output)

        return output, h_n, c_n

    def decode(self, h_n, c_n, lens,):
        max_len = int(max(lens))

        self.batch_size = encoder_outputs.size()[0]

        inp = self.dummy_decoder_input(batch_first=False)
        final_reconstructed = self.dummy_output(max_len, batch_first=False)

        for i in range(max_len):
            rnn_output, h_n, c_n = self.single_step_deocde(
                inp, encoder_outputs, h_n, c_n
            )

            final_reconstructed[i] = rnn_output
            inp = rnn_output

        return final_reconstructed.permute(1, 0, 2)


if __name__ == "__main__":
    k = torch.rand(2, 6, 64, 64)
    print(k.shape)

    vae = YieldModel()

    print(vae(k))