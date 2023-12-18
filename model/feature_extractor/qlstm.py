from typing import Optional

import torch
import torch.nn as nn

import pennylane as qml
# https://github.com/rdisipio/qlstm/blob/main/qlstm_pennylane.py

class QLSTM(nn.Module):
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=4,
                n_qlayers=1,
                batch_first=True,
                return_sequences=False, 
                return_state=False,
                backend="default.qubit"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        #self.dev = qml.device("default.qubit", wires=self.n_qubits)
        #self.dev = qml.device('qiskit.basicaer', wires=self.n_qubits)
        #self.dev = qml.device('qiskit.ibm', wires=self.n_qubits)
        # use 'qiskit.ibmq' instead to run on hardware

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]
        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]
        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]
        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]
        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        #self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        device = 'cpu'
        x = x.to(device)
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(device)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size).to(device)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            h_t = h_t.to(device)
            x_t = x_t.to(device)
            v_t = torch.cat((h_t, x_t), dim=1).to(device)

            # match qubit dimension
            y_t = self.clayer_in(v_t).to(device)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t))).to(device)  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t))).to(device)  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t))).to(device)  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t))).to(device) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            c_t = c_t.to(device)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class QLSTMFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        stride: int,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_size)
        #self.conv = nn.Conv1d(
        #    in_channels=hidden_size,
        #    out_channels=hidden_size,
        #    kernel_size=stride,
        #    stride=stride,
        #    padding=1,
        #)
        self.height = hidden_size * (2 if bidirectional else 1)
        self.lstm = QLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_qlayers=num_layers,
            n_qubits=in_channels,
            batch_first=True,
        )
        self.out_chans = 1
        self.out_size = out_size
        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        #x = self.fc(x.transpose(1, 2))  # x: (batch_size, time_steps, hidden_size)
        #x = self.conv(x.transpose(1, 2))  # x: (batch_size, hidden_size, time_steps)
        
        if self.out_size is not None:
            x = x.unsqueeze(1)  # x: (batch_size, 1, in_channels, time_steps)
            x = self.pool(x)  # x: (batch_size, 1, in_channels, output_size)
            x = x.squeeze(1)  # x: (batch_size, in_channels, output_size)
        x = x.transpose(1, 2)  # x: (batch_size, output_size, in_channels)
        x = self.fc(x)  # x: (batch_size, output_size, hidden_size)
        x, _ = self.lstm(x)  # x: (batch_size, output_size, hidden_size * num_directions)
        x = x.transpose(1, 2)  # x: (batch_size, hidden_size * num_directions, output_size)
        x = x.unsqueeze(1)  # x: (batch_size, out_chans, hidden_size * num_directions, time_steps)
        return x
