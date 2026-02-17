import numpy as np
import torch
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from torch import nn, optim
import torch.nn.functional as f
from torch.nn import GRU
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
import mlflow

from src.dataloaders import get_dataloaders


NUM_CLASSES = 4


# processes 320x320 2 channel input into a 128-long 1x1 feature vector
class CNN(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 8, 5, 5).to(device)
        self.conv2 = nn.Conv2d(8, 32, 8, 8).to(device)
        self.conv3 = nn.Conv2d(32, 128, 8).to(device)


    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))

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
    mlflow.set_tracking_uri("file:../mlruns")

    with mlflow.start_run(run_name="weighted cross entropy loss"):
        # load model and data
        model = FPVTrickDetector(device)

        train_dl, val_dl = get_dataloaders()

        # training setup
        num_epochs = 200
        early_stop_patience = 5
        optimizer = optim.Adam(model.parameters())
        loss_weights = torch.tensor(
            [0.005012037173141494, 0.13020393102887085, 0.22478045542458383, 0.6400035763734039]
        ).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weights)
        scaler = torch.amp.GradScaler('cuda')

        best_val_loss = np.finfo(np.float16).max
        best_state_dict = model.state_dict()
        early_stop_counter = 0
        train_metric_step = 0

        # training
        for epoch in range(num_epochs):
            # validation
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
                mlflow.log_metric("val_loss", val_loss, step=train_metric_step)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = model.state_dict()
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter == early_stop_patience:
                    break

            # optimization
            model.train()
            for inputs, labels in iter(train_dl):
                optimizer.zero_grad()

                with torch.amp.autocast("cuda"):
                    preds = model(inputs.to(device))

                    # rearrange preds from [batch, step, class] to [batch, class, step]
                    # for CrossEntropyLoss
                    preds = torch.moveaxis(preds, 2, 1)

                    loss = criterion(preds, labels.to(device))

                    mlflow.log_metric("train_loss", loss.item(), step=train_metric_step)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_metric_step += 1

        # reset model to best state
        model.load_state_dict(best_state_dict)

        # compute and log metrics and artifacts
        model.eval()
        y_pred = torch.empty(0).to(device)
        y_true = torch.empty(0).to(device)
        with torch.no_grad():
            for inputs, labels in iter(val_dl):
                with torch.amp.autocast("cuda"):
                    preds = model(inputs.to(device))
                    preds = torch.argmax(preds, dim=2)
                    preds = torch.flatten(preds)

                    labels = torch.flatten(labels).to(device)

                    y_pred = torch.cat([y_pred, preds])
                    y_true = torch.cat([y_true, labels])
        y_pred = y_pred.to('cpu')
        y_true = y_true.to('cpu')

        labels = [0, 1, 2, 3]
        display_labels = ['none', 'roll', 'flip', 'spin']
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot()
        plt.title("Confusion matrix")
        cm_fig = disp.figure_
        mlflow.log_figure(cm_fig, "confusion_matrix.png")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall
            },
        )

        # log model
        inputs, _ = next(iter(val_dl))
        inputs = inputs.to(device)
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                signature = infer_signature(inputs, model(inputs))
        mlflow.pytorch.log_model(model, signature=signature)


if __name__ == "__main__":
    main()
