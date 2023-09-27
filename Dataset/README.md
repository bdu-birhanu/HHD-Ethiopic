## HHD-Ethiopic Dataset
This dataset, named "HHD-Ethiopic," is designed for ethiopic text-image recognition tasks. It contains a collection of historical handwritten Manuscripts in the Ethiopic script. The dataset is intended to facilitate research and development for Ethiopic text-image recognition.

### Dataset Details
Size: 79,684 <br>
Training Set: 57,374 <br>
Test Set I (IID): 6,375 images (randomly drawn from the training set) <br>
Test Set II (OOD): 15,935 images (specifically from manuscripts dated in the 18th century) <br>
Validation Set: 10% of the training set, randomly drawn <br>
Number of unique Ethiopic characters :306

### Dataset Formats
The HHD-Ethiopic dataset is stored in two different formats to accommodate different use cases:

#### Raw Image and Ground-truth Text:

This format includes the original images and their corresponding ground-truth text.
The dataset is structured as raw images (.png) accompanied by a CSV file that contains the file names of the images and their respective ground-truth text.

#### Numpy Format (Images and Ground-truth Text):

In this format, both the images and the ground-truth text are stored in a convenient numpy format.
The dataset provides pre-processed numpy arrays that can be directly used for training and testing models.

#### Accessing the Dataset
You can access both formats of the HHD-Ethiopic dataset from the Hugging Face model hub, which serves as a reliable source for various datasets and models. The dataset is available at the following link:[HHD-Ethiopic](https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic)

Additionally, the dataset has been assigned a DOI for easier citation and reference: DOI: [10.57967/hf/0691](https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic)

### Human Level Performance Metadata
In the HHd-Ethiopic dataset, we have included metadata regarding the human-level performance predicted by individuals for the test sets. This metadata provides insights into the expected performance-level that humans can achieve in historical Ethiopic text-image recognition tasks.

#### Test Set I  - Human-Level Performance
For test set I, a group of 9 individuals was presented with a random subset of the dataset. They were asked to perform Ethiopic text-image recognition and provide their best efforts to transcribe the handwritten texts. The results were collected and stored in a CSV file, [Test-I-human_performance](https://github.com/bdu-birhanu/HHD-Ethiopic/blob/main/Dataset/human-level-predictions/6375_new_all.csv) included in the dataset.

#### Test Set II - Human-Level Performance
Test set II which was prepared exclusively from Ethiopic historical handwritten documents dated in the 18th century. A different group  of 4 individuals was given this subset for evaluation.  The human-level performance predictions for this set are also stored in a separate CSV file, [Test-II_human_performance](https://github.com/bdu-birhanu/HHD-Ethiopic/blob/main/Dataset/human-level-predictions/15935_new_all.csv)

Please refer to the respective CSV files for detailed information on the human-level performance predictions. Each CSV file contains the necessary metadata, including the image filenames, groind-truth and the corresponding human-generated transcriptions.

If you would like to explore or analyze the human-level performance data further, please refer to the provided CSV files.



#### Citation
If you use the hhd-ethiopic dataset in your research, please consider citing it:

```
@misc {birhanu_2023,
  author = { {Birhanu H.Belay, Isabelle Guyon,Tadele Mengiste, Bezawork Tilahun, Marcus Liwicki, Tesfa Tegegne, Romain Egele },
	title  = { HHD-Ethiopic:A Historical Handwritten Dataset for Ethiopic OCR with Baseline Models and Human-level Performance (Revision 50c1e04) },
	year   = 2023,
	url    = { https://huggingface.co/datasets/OCR-Ethiopic/HHD-Ethiopic },
	doi    = { 10.57967/hf/0691 },
	publisher = { Hugging Face }
}
```

#### License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
