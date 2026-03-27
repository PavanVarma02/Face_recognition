# Dataset Preparation

Dataset Preparation
Three public datasets were used in this pipeline AT&T, IMFDB and IMDB-WIKI. Each dataset has a different structure, which requires different handling during preparation.

## AT&T Database of Faces
AT&T is the smallest and most controlled dataset among the three. It comes with a Training folder containing 40 subfolders named s1 to s40, where each subfolder represents one unique person and contains exactly 10 images. All images are stored in .pgm format which is a grayscale image format. Since there are no explicit identity labels provided, each subfolder name was used as the identity label and a ID was asigned to each, s1 gets ID 0, s10 gets ID 1, and so on up to s40.
The images in AT&T are taken under controlled conditions same background, consistent lighting and mostly front face orientation. This makes it the easiest dataset to work with and generally produces the highest accuracy scores among the three.

## IMFDB: Indian Movie Face Database
IMFDB is a real world dataset containing faces of Indian cinema celebrities. It contains 100 subfolders each named after a different celbrity. Unlike AT&T, the number of images per person is not fixed and varies significantly some celebrities have a few hundred images while others have more. Images are named in the format {folder_name}_{number}.jpg.
Since each folder directly represents one person, the folder name was used as the identity label and a numeric ID was assigned to each celebrity. However since many celebrities have hundreds of images, using all of them would make the pipeline extremely slow and in testing caused the system to run out of memory and crash. To handle this, a maximum of 20 images per person was used, selected randomly. This kept the dataset manageable while still providing enough variety for meaningful evaluation.

## IMDB-WIKI Dataset
IMDB-WIKI is the largest and most complx dataset among the three. Unlike AT&T and IMFDB where each subfolder directly represents one person, IMDB-WIKI has a completely different structure. It contains 10 subfolders named 00 to 09 and inside each subfolder are images of multiple different people with no per-person folder at all.
The only way to identify which person an image belongs to is through the filename itself. Each image follows a specific naming convention:
`nm0000200 _ rm691128320 _ 1955-5-17 _ 2013 .jpg` where each part before underscore represents IMDB Person ID, Photo ID, Birth Date, Photo Year
The first part before the underscore is the IMDB person ID (e.g. nm0000200). All images sharing the same IMDB ID belong to the same person.
To handle this structure, all images across all 10 subfolders were scanned and grouped by their IMDB person ID into a temporary folder. For example all images starting with  `nm0000200` were copied into a single folder named `nm0000200`. After processing all subfolders, a total of 1622 unique identities were found, but only 500 identities were taken into consideration. Each identity folder was then assigned a numeric ID from 0 to 499.
Similar to IMFDB, only 20 images per person were used to keep processing manageable.

## Train/Eval Split
Once all three datasets were organized into per-person folders, the images were split into two sets train and eval. The train set acts as the reference image, storing reference discriptors for each identity. The eval set acts as the test set, containing images used to test whether the model can correcty identify a person by comparing against the refernce.
The split was done with a 40/60 ratio 40% of images per person go to train and 60% go to eval. This ensures the assignment requirement is always satisfied: the number of eval images must always exceed the number of train images for every identity.
To ensure each model gets same dataset, `random.seed(42)` was used before any shuffling. This means running the script multiple times always produces the exact same split, making results consistent and comparable.
A label map was saved as `label_map.json` mapping each identity name to its numeric ID. This file is used by all subsequent steps in the pipeline to maintain consistent labeling across embedding extraction and evaluation.

## Example Label map and split dataset

```python
Label map
{
  "s1": 0,
  "s10": 1,
  "s11": 2,
  "s12": 3,
  "s13": 4,
  "s14": 5,
  "s15": 6,
  "s16": 7,
  "s17": 8,
  "s18": 9,
  "s19": 10,
  "s2": 11,
  "s20": 12,
  "s21": 13,
  "s22": 14,
  "s23": 15,
  "s24": 16,
  "s25": 17,
  "s26": 18,
  "s27": 19,
  "s28": 20,
  "s29": 21,
  "s3": 22,
  "s30": 23,
  "s31": 24,
  "s32": 25,
  "s33": 26,
  "s34": 27,
  "s35": 28,
  "s36": 29,
  "s37": 30,
  "s38": 31,
  "s39": 32,
  "s4": 33,
  "s40": 34,
  "s5": 35,
  "s6": 36,
  "s7": 37,
  "s8": 38,
  "s9": 39
}

Training
  Identities : 40  
  Train imgs : 40
  Eval  imgs : 120
  Eval > Train check : True


IMFDB FR dataset
  Identities : 100  
  Train imgs : 800
  Eval  imgs : 1200
  Eval > Train check : True

IMDB-WIKI: 1622 valid identities found
IMDB-WIKI: copied 500 identities to temp folder

_temp
  Identities : 500  
  Train imgs : 1981
  Eval  imgs : 3367
  Eval > Train check : True
```


# Face Descriptors Extraction

Once the dataset is split into train and eval sets the next step is extracting face descriptors. A face descriptor is a set of numbers that represents a face similar faces produce similar numbers and different faces produce different numbers. Three different methods were used to extract these descriptors.

Before passing any image to the model a basic preprocessing step is done. Since AT&T images are grayscale they are first converted to a 3 channel image because all models require color images as input. The image is initially passed through the model at 320x320 size. If no face is found it is resized to 160x160 and tried again. Once descriptors are extracted they are saved as numpy arrays along with their identity labels and normalized using L2 norm.


## FaceAnalysis by InsightFace

FaceAnalysis from the InsightFace library handles the entire face recognition process in one place detection, alignment and feature extraction all happen internally.When an image is passed to FaceAnalysis the first thing it does is detect faces using RetinaFace. RetinaFace is a CNN based face detector that works reliably across different face sizes, lighting conditions and angles.

After detection the model finds landmark points on the face 106 points in 2D and 68 points in 3D. These landmarks mark specific locations like eye corners, nose tip and mouth corners.
Once landmarks are found the face is aligned. Five specific points both eye centers, nose tip and both mouth corners are used to compute a transformation that brings the face to a fixed standard position in a 112x112 image. This transformation is then applied to all pixels in the image. Alignment is important because recognition works by comparing pixel differences between facial features if the face is not in a standard position the model gets confused even for the same person.

After alignment the ArcFace model extracts features using a ResNet50 architecture. ArcFace was trained on 600,000 identities and produces 512 numbers representing the face descriptors.


## LBP (Local Binary Patterns)

LBP is a classical method that does not require any pretrained model or GPU. It works purely on pixel mathematics.

For each pixel in the image its value is compared with its 8 surrounding neighbors. If a neighbor is greater than or equal to the center pixel it counts as 1 otherwise 0. Summing these gives a value between 0 and 8 for each pixel.

The image is divided into small grid cells and a histogram of these values is built for each cell. All histograms are joined together to form the final descriptor of 576 numbers.

LBP works reasonably well on controlled datasets like AT&T because those images are already cropped, frontal and have consistent lighting. It struggles on real world datasets like IMFDB and IMDB-WIKI because it has no face detection or alignment step the entire image is treated as a face. Any variation in pose, lighting or scale significantly affects its performance.


## Face ecognition Library

The face_recognition library works similarly to FaceAnalysis but uses different methods. Face detection is done using HOG (Histogram of Oriented Gradients) which looks fpr edge patterns in the image to find faces. After detection 68 landmark points are identified and 5 key points are used to align the face the same way as FaceAnalysis. Feature extraction is done using a ResNet network producing 128 numbers per face.

The main difference from FaceAnalysis is that HOG based detection is less robust than RetinaFace it works well on frontal faces but struggles with side profiles, small faces and poor lighting, which leads to more images being skipped on challenging datasets.


# Evaluation


The metric used for this evaluation pipeline are Rank1 accuracy, Top accuracy, True Acceptane Rate (TAR), False Acceptance Rate (FAR)  and AUC of the ROC curve.
 ## Rank-1 Accuracy
Rank 1 accuracy is very strict metric, it tells us how often model is able to correctly predict on very first try with no second chances.
 ## TOP 5 Accuracy
It tells us how often the ground truth was suggested by the model even if it was not on top.
## True Acceptance rate (TAR)
It tells us out of all genuine pair (same person) how many are correctly accepted by the system at a given threshold. High TAR means model rarely misses real match. 
Genuine pair means same person image from reference and test.
## False Acceptance rate (FAR)
 It tells us that out of all imposter pairs, how many of them are correctly accepted by the model at a given threshold. A low FAR means model rarely confuses between different people.
Imposter pairs means different person from reference and test images.

## AUC of ROC curve 
It measures the ability of model to separate genuine pairs from impostor pairs across all possible thresholds.

## Results
Results of differnet model is presented below, images of AUC can be found in results directory. 
### LBP model
| Dataset | Rank-1 | Top-5 | TAR | FAR | AUC |
|---|---|---|---|---|---|
| AT&T | 75.83 % | 88.33% | 65.0% | 1.11.0% |95.08%|
| IMFDB | 7.5% | 19.0% | 2.51% | 1.0% | 53.91% |
| IMDB-WIKI | 2.55% | 4.75% | 2.11% | 1.0% | 52.78%|


### Face recognition model
| Dataset | Rank-1 | Top-5 | TAR | FAR | AUC |
|---|---|---|---|---|---|
| AT&T | 95.7% | 96.6% | 100.0% | 100.0% |100.0%|
| IMFDB | 60.19% | 80.58% | 44.22% | 1.01% | 93.76% |
| IMDB-WIKI | 51.87% | 57.26% | 37.05% | 1.0% | 72.92%|

### Face Analysis model
| Dataset | Rank-1 | Top-5 | TAR | FAR | AUC |
|---|---|---|---|---|---|
| AT&T | 100.0% | 100.0% | 100.0% | 100.0% |100.0%|
| IMFDB | 90.41% | 97.26% | 76.47% | 1.0% | 96.74% |
| IMDB-WIKI | 60.85% | 63.17% | 40.83% | 1.0% | 71.83%|
