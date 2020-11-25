import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import librosa
import tifffile


def get_df_from_files(path):
	"""
	Create a dataframe with the columns 'fn' and 'label' that includes the file paths and the corresponding labels, repsectively
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
		keywords_labels.append([keywords[i]]*keywords_freq_labels[i])
	
	keywords_files = [y for x in keywords_files for y in x]
	keywords_labels = [y for x in keywords_labels for y in x]
	
	keywords_paths = []
	for i in range(len(keywords_files)):
		keywords_paths.append(path + keywords_labels[i] + '/' + keywords_files[i])
		
	keywords_data = {'fn': keywords_paths, 'label': keywords_labels}
	keywords_df = pd.DataFrame(data=keywords_data)
	
	return keywords_df


def get_melspectrogram_from_path(path):
	"""
	Create a melspectrogram from an audiofile.
	"""
	
	y, sr = librosa.load(path)
   
	mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
	mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
	mel_spect = pad_db(mel_spect, (130,130))
	
	#normalize
	mel_spect = (mel_spect + 80) / 80
	 
	return mel_spect


def pad_db(array, reference_shape):
	"""
	array: Array to be padded
	reference_shape: tuple of size of ndarray to create
	"""
	result = np.full(reference_shape, -80.)
	result[:array.shape[0], :array.shape[1]] = array
	return result


def get_image_for_label(label):
	"""
	Show all images from a given class.
	"""

	files = train[train['label'] == label]['fn'].index.to_list()
	fig, ax = plt.subplots(len(files), figsize=(15,15))
	for i in range(len(files)):
		ax[i].imshow(get_melspectrogram_from_path(paths[files[i]]))
		ax[i].axis('off')
		ax[i].title.set_text(label + ' (' + str(files[i]) + ')')
	plt.tight_layout()


def download_images(paths):
	"""
	Download melspectrograms into the folder './spectrograms/'
	"""

	for i in range(len(paths)):
		output_file = paths[i].split('/')[-1].split('.wav')[0]
		output_file = output_file + '.tif'
		tifffile.imwrite('./spectrograms/' + output_file, get_melspectrogram_from_path(paths[i]))
		