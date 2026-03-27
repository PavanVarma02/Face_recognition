This repository features the evaluation pipeline for three different dataset : AT&T, IMFDB and IMDB-WIKI. The evaluation pipeline can be re run on the dataset by running three different file : dataset_prep.py, face_descriptors.py, evaluate.py.
The dataset can be downloaded using below links:
- AT&T : [Download](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/)
- IMFDB:[Download](https://www.kaggle.com/datasets/anirudhsimhachalam/indian-movie)
- IMDB-WIKI : [Download](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki)
  
The pipeline performance is evaluated over five metrics: Rank-1 Accuracy, Top 5 Accuracy, True Acceptance Rate (TAR), False Acceptance Rate (FAR) and AUC of the ROC curve.

## How to run the code
Make virutal env.
Grab the code from GitHub repository
```python
git clone https://github.com/PavanVarma02/Face_recognition.git
```
Install following packages
```python
pip install numpy opencv-python scikit-learn matplotlib
pip install insightface onnxruntime
pip install face_recognition
```
### Step: 1   Dataset Preparation

Before running the `dataset_prep.py` file, in the main section of code set the path for each downloaded dataset directory, the path where you want to store the preapred dataset, and the name of the dataset folder.
Run the following python file
```python
dataset_prep.py
```
### Step: 2   Face desciptors

On executing dataset preparation file it will create seprate directory containing dataset, before running the `face_desciptors.py` file in tis main section specify the director containing the prepared dataset, also specify the directory for descriptor output.
Run the following python file
```python
{model}_descriptors.py
```
Since we here we have compared 3 diffferent model, to get descriptors for each model run different descriptor file.

#### For FaceAnalysis model
```python
FA_descriptors.py
```

#### For LBP model
```python
LBP_descriptors.py
```
#### For Face recognition model
```python
FR_descriptors.py
```
### Step:3 evaluation 

Upon running the descriptors python file, the descriptors for each dataset will be created, same as above in main section of `evaluate.py` file specify the directory containing the desciptors, also mention the result folder for the output, following this run below python file
```python
evaluate.py
```

