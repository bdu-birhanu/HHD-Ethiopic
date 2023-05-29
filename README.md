
# HHD-Ethiopic 

A text-line level historical handwritten Ethiopic OCR Dataset

## Overview
This repository contains dataset called HHD-Ethiopic, and baselines models and human-level performance for benchmarking Historical Handwritten Ethiopic text-image recognition. HHD-Ethiopic is a text-line level historical handwritten Ethiopic OCR Dataset specifically designed for historical handwritten Ethiopic text-image recognition tasks. 

## Dataset Details
The HHD-Ethiopic dataset consists of ~80k text-line images extracted from $18^{th}$ to $20^{th}$ centuries historical handwritten Ethiopic manuscripts. Each text-line image is accompanied by its ground-truth text transcription. The dataset can be directly downloaded from Hugging Face [HHD-Ethiopic Dataset](https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic) and/or Zenodo [HHD-Ethiopic Dataset](https://zenodo.org/record/7978722).  Sample text-line images and their corresponding ground-truth text are shown below. For a more thorough tutorial about the dataset see [formats of the dataset](https://github.com/bdu-birhanu/HHD-Ethiopic/tree/main/Dataset)

| No. | Text-line Image | Ground-Truth Text |
|--|-------|------------------|
| [Image 1] |![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/a96171d0-6850-41ac-a960-2f8e6edeff57) | ወጽራኅየኒ፡ቦአ፡ቅድሜሁ፡ውስተ፡ዕዘኒሁ  |
| [Image 2] |![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/2e526959-3e94-4295-84d8-33dc44d478f8)  | ፍራስ፡እሳት፡ወጽሩዓን |
| [Image 3] |![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/9ede4ee9-724b-4327-9790-41cc0e28041d)   | ወአንሰ፡በብዝኃ፡አሀውዕ፡ቢተኩ |
| [Image 4] | ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/4de16d8d-47df-4c53-a73f-f2c454ad8853) | ወአድኅነከ፡ይትፌሥሑ።  |

## Getting Started
In the current implementation, the HHD-Ethiopic dataset is used for training and testining the baesline models. Download the dataset from the above link.

After downloading HHD-Ethiopic, install the requirements, to demonstarte we just use the data stored in numpy format:
 ```markdown
pip install -r requirements.txt
  ```
  
To Train the model from scratch
```markdown
$ python3 train_model_plain_CTC.py
```

To Prediction/test
```markdown
$ python3 test_model_plain_CTC.py
``` 
Alternatively you can test the model by runing the google colab code directly as below.
### Code Testing

You can access the code demonstration in Google Colab  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([paste_the_copied_notebook_URL_here](https://github.com/bdu-birhanu/HHD-Ethiopic/blob/main/demo_Plain_CTC.ipynb)). Sample results and Character Error Rate (CER) per line are shown below:
| <sub>Image</sub>| <sub>Ground-truth</sub> | <sub> Prediction </sub>| <sub> Edit Distance</sub> | <sub>CER/Line (100%) </sub>|
|-------|--------------|------------|---------------|----------|
|<sub> ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/dd64c5d2-c9d2-4928-bd61-9f9d8c86a7b1) </sub>| <sub> ሰፉሐከ፡የማነከ፡ወውሕጠቶሙ፡ምድር። </sub>|  <sub> ሰፉሕከ፡የማነከ፡ወውሕጠቶሙ፡ምድ። </sub>| 2 | 9 |
| ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/703410fe-635c-434e-9a89-7a4144f5d4c9) | <sub> ምድር፡ይኔጽር፡ዘሀሎ፡በየብስ፡</sub>   |  <sub> ምድር፡ይኔጽር፡ዘሀሎ፡በየብስ፡ </sub> | 1 | 5|
| ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/46384b40-1112-42d2-b608-ba01298efa39) |<sub> ለብሒረ፡ኢትየጵያ </sub> |  <sub>  አብሔረ፡ኢትዮጵያ </sub> | 4| 40|
| ![download](https://github.com/bdu-birhanu/HHD-Ethiopic/assets/35142364/9baf0493-1fdf-4817-aa2b-700a688ee90e) | <sub>ዓገሠ።በዝሕማም፡መሥጋ፡</sub> |  <sub>  ዓገሠ።በዝሕማም፡በሥጋ፡ </sub>| 2| 20|


            
## Feedbacks
We welcome contributions and feedback from the research community to further enhance the HHD-Ethiopic dataset and code. If you have any suggestions, bug reports, or improvements, please feel free to send them via email: ethiopic.dataset@gmail.com



## Acknowledgments
We would like to express our gratitude to the Ethiopian National Archive and Library Agency (ENALA) for providing access to the historical handwritten documents used in creating the HHD-Ethiopic dataset. We also acknowledge the support and contributions of the annotators who, making this dataset possible.


## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.


