import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class Spectrogram_Dataset_Augmentation(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.flow = naf.Sequential([nas.FrequencyMaskingAug(), nas.TimeMaskingAug()])
        if augment:
            self.augment = True

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        self.image = tifffile.imread('spectrograms/' + self.paths[index])
        if self.augment:
            apply_augment = np.random.choice(4, 1)
            if apply_augment == 1:
                self.image = self.flow.augment(self.image)
        self.image = torch.from_numpy(self.image)
        self.label = self.labels[index]
        sample = (self.image, self.label)
        return sample


def train_model_finetuning(net, net_name: str, train_ds, val_ds, batch_size: int, num_epochs: int, device,
                           save_model=True, early_stopping_patience=8):
    """
    :param net: instantiated model class
    :param net_name: name of the model
    :param train_ds: Dataset containing the training examples
    :param val_ds: Dataset containing the validation examples
    :param batch_size: mini-batch size
    :param num_epochs: number of epochs the model is trained
    :param device: device on which the model is trained
    :param save_model: set to False if weights of the model at the moment of the highest accuracy should be saved
    :param early_stopping_patience: number of epochs that accuracy does not increase before stopping the training
    :return: train_epoch_loss, valid_epoch_loss, acc_epoch as well as the plotted learning curve
    """

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    loss = nn.CrossEntropyLoss()

    train_epoch_loss, valid_epoch_loss, acc_epoch = [], [], []
    best_val_acc = 0

    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)
    early_stopping_count = 0

    for epoch in range(num_epochs):

        net.train()
        correct = 0

        train_batch_loss, valid_batch_loss = [], []

        for batch in train_dl:
            X_train = batch[0].to(dtype=torch.float)
            X_train = np.repeat(X_train[:, np.newaxis, :, :], 3,
                                axis=1).cuda()  # copies values to second and third channel
            y_train = batch[1].to(device)
            optimizer.zero_grad()
            preds = net(X_train)
            l = loss(preds, y_train)
            l.backward()
            optimizer.step()

            train_batch_loss.append(l.detach().item())

        train_epoch_loss.append(np.mean(train_batch_loss))

        net.eval()
        with torch.no_grad():
            for batch in val_dl:
                X_val = batch[0].to(dtype=torch.float)
                X_val = np.repeat(X_val[:, np.newaxis, :, :], 3, axis=1).cuda()
                y_val = batch[1].to(device)
                val_preds = net(X_val)
                valid_batch_loss.append(loss(val_preds, y_val).detach().item())
                predicted = torch.max(val_preds, 1)[1]
                correct += (predicted == y_val).sum()
                correct = correct.float()

            correct_epoch = (correct / len(val_dl.dataset))
            acc_epoch.append(correct_epoch)

        scheduler.step(correct_epoch)

        valid_epoch_loss.append(np.mean(valid_batch_loss))

        print(
            f'Ep: {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_loss[-1]:.4f} | '
            f'Val Loss: {valid_epoch_loss[-1]:.4f} | '
            f'Val Acc: {correct_epoch:.4f}')

        if save_model:
            if correct_epoch >= best_val_acc:
                print(
                    f'Validation accuracy has improved from {best_val_acc:.4f} to {correct_epoch:.4f}. Saving model...')
                torch.save(net.state_dict(),
                           f'/content/drive/MyDrive/Kaggle/GIZ_NLP_Agricultural_Keyword_Spotter/models/best_model_zindi'
                           f'_{net_name}_23-11.pt'.format(net_name))

        if correct_epoch >= best_val_acc:
            best_val_acc = correct_epoch
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count >= early_stopping_patience:
            print("Early Stopping")
            break

    torch.save(net.state_dict(),
               f'/content/drive/MyDrive/Kaggle/GIZ_NLP_Agricultural_Keyword_Spotter/models/last_model_zindi_{net_name}'
               f'_23-11.pt'.format(net_name))

    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    plot_learning_curve((epoch + 1), train_epoch_loss, valid_epoch_loss, acc_epoch, 'Loss/Accuracy')

    return train_epoch_loss, valid_epoch_loss, acc_epoch


def plot_learning_curve(num_epochs, train_loss, valid_loss, acc, y_label):
    plt.plot(range(num_epochs), train_loss, label='train loss')
    plt.plot(range(num_epochs), valid_loss, label='valid loss')
    plt.plot(range(num_epochs), acc, label='valid accuracy')
    plt.xticks(range(num_epochs), range(1, num_epochs + 1))
    plt.ylabel(y_label)
    plt.xlabel('Epochs')
    plt.legend()
