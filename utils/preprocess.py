import os
import pickle
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt


def map_labels(file, label: str):
    """
    :param file: Pandas DataFrame
    :param label: name of the column that holds the labels
    :return label_mapping: dictionary with the mapping between categorical labels and numeric labels
    :return label_numeric_mapping: dictionary with the mapping between numeric labels and categorical labels
    """

    labels_list = list(file[label].unique())

    labels_mapping = dict()
    i = 0
    for label in labels_list:
        if label not in labels_mapping:
            labels_mapping[label] = i
            i += 1

    labels_numeric_mapping = dict()
    for i in range(len(labels_list)):
        labels_numeric_mapping[i] = labels_list[i]

    return [labels_mapping, labels_numeric_mapping]


def save_as_pickle(path: str, file):
    """
    :param path: path to which the pickle is be saved
    :param file: file which is saved as pickle
    """
    f = open(path + '.pkl', 'wb')
    pickle.dump(file, f)
    f.close()


def get_df_from_file(path: str):
    """
    :param path: path to table-style file
    :return: DataFrame object
    """
    keywords = os.listdir(path)

    if '.DS_Store' in keywords:
        keywords.remove('.DS_Store')

    keywords_files = []
    for label in keywords:
        keywords_files.append(os.listdir(path + label))

    keywords_freq_labels = [len(x) for x in keywords_files]

    keywords_labels = []
    for i in range(len(keywords)):
        keywords_labels.append([keywords[i]] * keywords_freq_labels[i])

    keywords_files = [y for x in keywords_files for y in x]
    keywords_labels = [y for x in keywords_labels for y in x]

    keywords_paths = []
    for i in range(len(keywords_files)):
        keywords_paths.append(path + keywords_labels[i] + '/' + keywords_files[i])

    keywords_data = {'fn': keywords_paths, 'label': keywords_labels}
    keywords_df = pd.DataFrame(data=keywords_data)

    return keywords_df


def get_melspectrogram(path):
    """
    :param path: path of audio file
    :return: melspectrogram padded to 130x130 pixels
    """
    y, sr = librosa.load(path)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    mel_spect = mel_spect[:, :130]  # two images have shape of 128,135 and 128,137 -- crop!
    mel_spect = pad_db(mel_spect, (130, 130))
    mel_spect = (mel_spect + 80) / 80
    return mel_spect


def pad_db(input_matrix, reference_shape):
    """
    :param input_matrix: matrix to be padded
    :param reference_shape: shape of matrix that is returned
    :return: padded matrix of shape reference_shape
    """
    result = np.full(reference_shape, -80.)
    result[:input_matrix.shape[0], :input_matrix.shape[1]] = input_matrix
    return result


def get_image_for_label(dataframe, label: str):
    """
    :param dataframe: dataframe with audio file paths
    :param label: class for which all images are shown
    :return: melspectrograms of a given class
    """
    files = dataframe[dataframe['label'] == label]['fn'].index.to_list()
    paths = list(dataframe.fn)
    fig, ax = plt.subplots(len(files), figsize=(15, 15))
    for i in range(len(files)):
        ax[i].imshow(get_melspectrogram(paths[files[i]]))
        ax[i].axis('off')
        ax[i].title.set_text(label + ' (' + str(files[i]) + ')')
    plt.tight_layout()
