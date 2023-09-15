
# HHD-Ethiopic 

A text-line level historical handwritten Ethiopic OCR Dataset

## Overview
This repository contains a dataset called HHD-Ethiopic, and baselines models and human-level performance for benchmarking Historical Handwritten Ethiopic text-image recognition. HHD-Ethiopic is a text-line level historical handwritten Ethiopic OCR Dataset specifically designed for historical handwritten Ethiopic text-image recognition tasks. 

## Dataset Details
The HHD-Ethiopic dataset consists of ~80k text-line images extracted from $18^{th}$ to $20^{th}$ centuries of historical handwritten Ethiopic manuscripts. Each text-line image is accompanied by its ground-truth text transcription. The dataset can be directly downloaded from Hugging Face [HHD-Ethiopic Dataset](https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic) and/or Zenodo [HHD-Ethiopic Dataset](https://zenodo.org/record/7978722).  Additional synthetically generated Ethiopic text-line images and their corresponding ground truth texts are available from [this link](https://drive.google.com/file/d/1fAPrAp4Hu8zEqs5XLV5dMtkXjyNGfMzg/view?usp=drive_link). 

Sample text-line images and their corresponding ground-truth text are shown below. For a more thorough tutorial about the dataset see [formats of the dataset](https://github.com/bdu-birhanu/HHD-Ethiopic/tree/main/Dataset)

| No. | Text-line Image | Ground-Truth Text |
|--|-------|------------------|
| [Image 1] |![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/a96171d0-6850-41ac-a960-2f8e6edeff57) | ወጽራኅየኒ፡ቦአ፡ቅድሜሁ፡ውስተ፡ዕዘኒሁ  |
| [Image 2] |![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/2e526959-3e94-4295-84d8-33dc44d478f8)  | ፍራስ፡እሳት፡ወጽሩዓን |
| [Image 3] |![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/9ede4ee9-724b-4327-9790-41cc0e28041d)   | ወአንሰ፡በብዝኃ፡አሀውዕ፡ቢተኩ |
| [Image 4] | ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/4de16d8d-47df-4c53-a73f-f2c454ad8853) | ወአድኅነከ፡ይትፌሥሑ።  |

## Getting Started
In the current implementation, the NumPy format of the HHD-Ethiopic dataset is used for training and testing the baseline models. Download the dataset.

After downloading HHD-Ethiopic, install the requirements, to demonstrate we just used the [Train data](https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic/blob/main/train/train_numpy.zip) and [Test data ](https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic/blob/main/test/test_rand/test_rand_numpy.zip) stored in numpy format.  To train and test all baseline models, please use [all source codes](https://github.com/bdu-birhanu/HHD-Ethiopic/tree/main/src/all_code) link.
 ```markdown
pip install -r requirements.txt
  ```
  
To Train the model from scratch
```markdown
$ python3 train_model_plain_CTC.py
```
Alternatively, you can also run  the training code demonstration in Google Colab directly [![68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/c13c4086-8278-47d4-8bde-ef19d3204439)](https://github.com/bdu-birhanu/HHD-Ethiopic/blob/main/train_HPopt_Attn_CTC.ipynb).

To Prediction/test
```markdown
$ python3 test_model_plain_CTC.py
``` 
Alternatively, you can also run the testing code demonstration in Google Colab directly [![68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/c13c4086-8278-47d4-8bde-ef19d3204439)](https://github.com/bdu-birhanu/HHD-Ethiopic/blob/main/Test_HPopt-Attn-CTC.ipynb).<br>

***Please note that the two Colab demos provided here are **the HPopt-Attn-CTC **implementation** as a** sample demo.**
### Sample testing results
Sample results and Character Error Rate (CER) per line are shown below:
| <sub>Image</sub>| <sub>Ground-truth</sub> | <sub> Prediction </sub>| <sub> Edit Distance</sub> | <sub>CER/Line (100%) </sub>|
|-------|--------------|------------|---------------|----------|
|<sub> ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/dd64c5d2-c9d2-4928-bd61-9f9d8c86a7b1) </sub>| <sub> ሰፉሐከ፡የማነከ፡ወውሕጠቶሙ፡ምድር። </sub>|  <sub> ሰፉሕከ፡የማነከ፡ወውሕጠቶሙ፡ምድ። </sub>| 2 | 9 |
| ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/703410fe-635c-434e-9a89-7a4144f5d4c9) | <sub> ምድር፡ይኔጽር፡ዘሀሎ፡በየብስ፡</sub>   |  <sub> ምድር፡ይኔጽር፡ዘሀሎ፡በየብስ፡ </sub> | 1 | 5|
| ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/46384b40-1112-42d2-b608-ba01298efa39) |<sub> ለብሔረ፡ኢትዮጵያ </sub> |  <sub> አብሒረ፡ኢትየጵያ  </sub> | 4| 40|
| ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/9baf0493-1fdf-4817-aa2b-700a688ee90e) | <sub>ዓገሠ።በዝሕማም፡መሥጋ፡</sub> |  <sub>  ዓገሠ።በዝሕማም፡በሥጋ፡ </sub>| 2| 20|


            
### Feedbacks
We welcome contributions and feedback from the research community to further enhance the HHD-Ethiopic dataset and code. If you have any suggestions, please feel free to send them via email: ethiopic.dataset@gmail.com



### Acknowledgments
We would like to express our gratitude to the Ethiopian National Archive and Library Agency (ENALA) for providing access to the historical handwritten documents used in creating the HHD-Ethiopic dataset. We are also grateful to ICT4D research center, Bahir Dar Institute of Technology, and ChaLearn for their funding. Furthermore, we would like to acknowledge the support and contributions of the annotators who made this dataset possible.


### License
![cc](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/49b9e794-f526-4f85-96c0-30b842c9abd0)This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.


