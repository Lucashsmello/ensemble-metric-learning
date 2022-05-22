from torch import nn


class NetArchModel(nn.Module):
    def __init__(self, input_size=6100, num_outputs=8, num_inputs_channels=1, activation_function=nn.PReLU()):
        super().__init__()
        self.convnet = nn.Sequential(  # input is (n_batches, 1, 6100)
            nn.Conv1d(num_inputs_channels, 16, 5), activation_function,  # 6100 -> 6096
            nn.Dropout(0.2),
            nn.MaxPool1d(4, stride=4),  # 6096 -> 1524
            nn.Conv1d(16, 32, 5), activation_function,  # 1524 -> 1520
            nn.Dropout(0.2),
            nn.MaxPool1d(4, stride=4),  # 1520 -> 380
            nn.Conv1d(32, 64, 5), activation_function,  # 380 -> 376
            nn.Dropout(0.2),
            nn.MaxPool1d(4, stride=4),  # 376 -> 94
            nn.Flatten(),
        )

        n = (input_size - 4)//4
        n = (n - 4)//4
        n = (n - 4)//4
        self.fc = nn.Sequential(nn.Linear(64 * n, 192),
                                activation_function,
                                nn.Linear(192, num_outputs)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = self.fc(output)
        return output
