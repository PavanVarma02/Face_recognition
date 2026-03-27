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

# Face Descriptors

