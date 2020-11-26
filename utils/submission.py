import torch
from torch.utils.data import Dataset


class Spectrogram_Dataset_Submission(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        self.image = torch.from_numpy(tifffile.imread(
            'test_spectrograms/' + self.paths[index]))
        return self.image


def test_model(net, test_dl, device):
    """
    :param net: the pretrained model
    :param test_dl: the test DataLoader
    :param device: the device on which the testing is performed
    :return: the predictions of the test set
    """

    net = net.to(device)
    net.eval()
    preds = []

    with torch.no_grad():
        for batch in test_dl:
            batch = batch.to(dtype=torch.float)
            batch = np.repeat(batch[:, np.newaxis, :, :], 3, axis=1).cuda()
            batch_preds = net(batch)
            preds.append(batch_preds)

    return preds


def preds_to_dataframe(preds, test_paths, column_order, labels_numeric_mapping):
    """
    :param preds: test predictions
    :param test_paths: paths of the audio files in the test set
    :param column_order: order of columns in SampleSubmission.csv
    :param labels_numeric_mapping: mapping of classes to numeric labels (as defined when preprocessing the training set)
    :return: a dataframe containing the probability for each class per example
    """
    preds = [p.cpu().numpy() for batch in preds for p in batch]
    preds_softmax = []
    for p in preds:
        p_softmax = softmax(p)
        preds_softmax.append(p_softmax)
    preds_softmax = np.array(preds_softmax)
    df = pd.DataFrame(data=preds_softmax)
    df['fn'] = test_paths
    df = df[column_order]
    df = df.rename(columns=labels_numeric_mapping)
    return df


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()
