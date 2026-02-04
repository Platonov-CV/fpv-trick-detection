import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import GRU

from src.dataloaders import get_dataloaders


NUM_CLASSES = 8


# processes 320x320 2 channel input into a 128-long 1x1 feature vector
class CNN(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 8, 5, 5).to(device)
        self.conv2 = nn.Conv2d(8, 32, 8, 8).to(device)
        self.conv3 = nn.Conv2d(32, 128, 8).to(device)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        return x


class FPVTrickDetector(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.cnn = CNN(device=device)

        self.rnn = GRU(
            input_size=128, hidden_size=128, num_layers=1, batch_first=True,
            bidirectional=True, device=device
        )

        self.fc = nn.Linear(in_features=256, out_features=NUM_CLASSES, device=device)

        # self.half()


    def forward(self, x):
        # reshape input from (B, T, C, H, W) into (B * T, C, H, W) for CNN
        # B - batch
        # T - frame
        # C - channels
        # H, W - flow dimensions
        b, t, c, h, w = x.shape
        x = torch.reshape(x, (b * t, c, h, w))

        x = self.cnn(x)

        # reshape from (B * T, F) to (B, T, F) for RNN
        # F - feature
        x = torch.reshape(x, (b, t, 128))

        x, _ = self.rnn(x)

        x = self.fc(x)

        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FPVTrickDetector(device)

    train_dl, val_dl = get_dataloaders()

    num_epochs = 200
    early_stop_patience = 5
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = np.finfo(np.float16).max
    best_state_dict = model.state_dict()
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in iter(train_dl):
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                preds = model(inputs.to(device))

                # rearrange preds from [batch, step, class] to [batch, class, step]
                # for CrossEntropyLoss
                preds = torch.moveaxis(preds, 2, 1)

                loss = criterion(preds, labels.to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 5 != 0:
            continue
        model.eval()
        with torch.no_grad():
            val_losses = []
            for inputs, labels in iter(val_dl):
                with torch.amp.autocast("cuda"):
                    preds = model(inputs.to(device))

                    # rearrange preds from [batch, step, class] to [batch, class, step]
                    # for CrossEntropyLoss
                    preds = torch.moveaxis(preds, 2, 1)

                    val_losses.append(criterion(preds, labels.to(device)).item())

            val_loss = sum(val_losses) / len(val_losses)
            print('Epoch: ', epoch, ', val_loss: ', val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter == early_stop_patience:
                break

    torch.save(best_state_dict, "../results/model.pt")


if __name__ == "__main__":
    main()
