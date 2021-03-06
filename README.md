# Zindi Data Science Challenge: Agricultural Keyword Spotter

This project is part of the [GIZ NLP Agricultural Keyword Spotter](https://zindi.africa/competitions/giz-nlp-agricultural-keyword-spotter) challenge by Zindi. 

The goal of this challenge is to classify keywords in Luganda and English in audioclips. These keywords relate to crops, diseases, fertilizers, and other agricultural topics. The keyword classifier is then used by Makerere University to  build a speech recognition model in order to automatically monitor radio prgrams in Uganda for agriculture-related information. Since in Uganda, radio programmes are essential to deliver information to rural communities, these programmes can bring valuable insights into the Ugandan agriculture sector to researchers, governments and other decision makers.

<p align="center">
  <img width="800" height="300" src="https://github.com/HeleneFabia/keyword-spotter/blob/main/images/weat.jpg">
</p>

***

#### The dataset

The data consists of around 4000 audio files, each containing one of 193 spoken keywords in English or Luganda. What poses a challenge here is that for some keywords there are only a handful of audio files in the training set.

After doing some research about how best to handle auido data, I decided to extract features from the audio files by creating mel spectrograms and treat the problem at hand as an image classification problem. For more detail about mel spectrograms and audio classification with CNNs, I recommend the following two articles here:
- https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
- https://medium.com/x8-the-ai-community/audio-classification-using-cnn-coding-example-f9cbd272269e

To get a grasp of the data, below are the mel the spectrograms for all audio files containing the keyword 'banana':

<p align="left">
  <img width="1000" height="250" src="https://github.com/HeleneFabia/keyword-spotter/blob/main/images/specs.png">
</p>

Since there are so few training examples, I decided to augment my data. As far as I understand it, you can augment audio data in two ways. Either you alter the audio itself (e.g. shift the pitch, stretch it, speed it up,...) or you augment the mel spectrograms. After some experiments, I settled for the second way using
a method called `SpecAugmet` (https://arxiv.org/pdf/1904.08779.pdf) via the `nlpaug` library (https://nlpaug.readthedocs.io/en/latest/index.html). Below are three augmentations of the first mel spectrograms from above. The first image shows Frequency Masking, the second Time Masking, and the third one combines both Frequency and Time Masking.

<p align="left">
  <img width=720" height="240" src="https://github.com/HeleneFabia/keyword-spotter/blob/main/images/augm.png">
</p>

***

#### The model

After looking into using CNNs for audio classification, I saw that using a transfer learning approach with CNNs pretrained on ImageNet works quite well for mel spectrograms:
- https://medium.com/@krishna_84429/audio-classification-using-transfer-learning-approach-912e6f7397bb 
- https://arxiv.org/pdf/2007.11154.pdf).

After some experiments involving multiple pretrained models in torchvision, using `densenet161`, `resnet50`, and  `googlenet` yielded the best validation accuracy: 0.8355, 0.7251, and 0.7898, respectively.

***

#### Submission

I ensembled the three models mentioned above in order to improve the model's ability to generalize. This solution got me to the 43rd place out of 222 participants in the challenge.

***

For more details, please view my notebooks about [preprocessing](https://github.com/HeleneFabia/keyword-spotter/blob/main/keyword_spotter_preprocessing.ipynb), [training](https://github.com/HeleneFabia/keyword-spotter/blob/main/keyword_spotter_training.ipynb), and [submitting](https://github.com/HeleneFabia/keyword-spotter/blob/main/keyword_spotter_submission.ipynb).
