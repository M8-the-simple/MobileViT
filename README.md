# Integration and evaluation of Transformer models for face recognition on edge devices


This project implements a modular face recognition pipeline that compares several embedding models (MobileViT, ConvNeXt, ViT Tiny, and InsightFace) using centroid-based classification.

Project has scripts for inter and intra-class testing. Using these distributions of similarity scores allows for generation of ROC and PR curves for data visualization. Centroids and similarity scores can be used to make a PCA analysis for determining the discriminative power of models. 

Goal of this project was to evaluate and integrate Transformer models for face recognition on edge devices, ViT Tiny was chosen for further optimization such as quantization and deployment in the final system for access control.
## System overview

Video/image &rarr; Face Detector &rarr; Image preprocessing &rarr; Feature extraction model &rarr; Classification (centroid-based) &rarr; Identified / Unknown

## 🚀 Features
- **Multi-Model Backend**: Support for multiple architectures via `config.yaml`.
- **Flexible Detection**: Integrated face detection using MTCNN or Haar Cascades.
- **Quantization Support**: Ability to run quantized versions of models for edge deployment.
- **Centroid Classification**: Uses identity centroids for efficient recognition and ease of adding new identities.
- **Analysis Tools**: PCA-based embedding visualization and centroid density analysis.

## 🛠 Tech Stack
- **Language**: Python 3.12+
- **Core Libraries**: OpenCV, PyTorch, NumPy, Pathlib, TorchAO, timm, InsightFace
- **Models**: [MobileViT](https://huggingface.co/timm/mobilevitv2_050.cvnets_in1k), [ConvNeXt](https://huggingface.co/gaunernst/convnext_nano.cosface_ms1mv3), [ViT Tiny](https://huggingface.co/gaunernst/vit_tiny_patch8_112.arcface_ms1mv3), [InsightFace](https://github.com/deepinsight/insightface) (via Hub)

## 🏃 Quick Start

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate       #Linux/macOS
# .venv\Scripts\activate        #Windows  

pip install -r requirements.txt
```

### 2. Configuration
Adjust `config.yaml` to select your active backend and set detection thresholds:
```yaml
system:
  backend: "ViT_Tiny" # Change to MobileViT, ConvNeXt, or InsightFace
```

### 3. Execution
If you have train and test videos you should put them in ```train_videos``` and ```test_videos``` folders and define the paths in the ```config.yaml``` file.

Run both:
```bash
python -m data.extract_frames train 
```
```bash
python -m data.extract_frames test 
```
After the images are created run:
```bash
python -m data.build_centroids
```
Now you can either run temporal analysis or live face recognition:
```bash
python -m pipeline --test
```
```bash
python -m pipeline 0
```



## 🚀 Quick Utilities

| Goal | Script | Basic command | Requirements
|------|--------|---------------|------------|
|Extract frames|data/extract_frames.py| python -m data.extract_frames \<mode>| None |
|Generate centroids|data/build_centroids.py| python -m data.build_centroids \<person>| Pictures in ```dataset/train``` |
|Create similarity scores|evaluation/tests/generate_labels.py| python -m evaluation.tests.generate_labels --all| Generated centroids |
|Generate ROC and PR curves|evaluation/tests/classification_evaluation.py| python -m evaluation.tests.classification_evaluation --all| Generated similarity scores |
|PCA analysis|src/analysis/pca.py| python -m src.analysis.pca --mode all| At least three centroids or one person with three embeddings

## 📁 Project Structure
- `data/`: Scripts for dataset preparation and frame extraction.
- `src/`: Core logic, model wrappers, and statistical analysis.
- `centroids/`: Precomputed embedding centroids for different models.
- `dataset/`: Extracted face images (generated).
- `evaluation/`: Testing suites and label generation.
- `config.yaml`: Global system and model configuration.

## Usage

The scripts should be executed from the project root directory. The recommended workflow consists of dataset preparation, centroid generation, similarity-score generation, evaluation, and recognition.

### 1. Preparing the dataset

Place training and testing videos in the corresponding directories:

```text
train_videos/
├── PersonName/
└── ...

test_videos/
├── PersonName/
└── ...
```


For frame extraction you should use the following commands:
```bash
python -m data.extract_frames train
python -m data.extract_frames test
```
You can also specify a person:
```bash
python -m data.extract_frames <mode> PersonName
```

If you would like to remove previously extracted frames you should use:
```bash
python -m data.extract_frames <mode> --remove
```

Script tries to detect a face on every fourth frame and if any of the detected faces has a confidence value higher than ```0.95``` the entire video frame is saved in the ```dataset/<mode>/PersonName/``` folder. 

### 2. Centroid generation
After preparing the training dataset, generate the centroids used for classification:
```bash
python -m data.build_centroids
```
The generated files are stored in a model-specific directory:
```text
centroids/<model_name>/
├── centroid_znacajke.npy
└── oznake.npy
```

### 3. Generating similarity scores

To evaluate the capability to separate positive cases and negative cases we generate similarity scores for all persons:
```bash
python -m evaluation.tests.generate_labels --all
```
Generate similarity scores for a specified identity:
```bash
python -m evaluation.tests.generate_labels PersonName
```
Limit the maximum amount of frames used:
```bash
python -m evaluation.tests.generate_labels PersonName --max 10
```

These similarity scores are stored in the ```y_score.npy``` and positive and negative cases are defined in ```y_true.npy``` both are stored in the following structure:
```text
evaluation/tests/similarities/<model_name>/
├── y_score.npy
└── y_true.npy
```

### 4. Calculating evaluation metrics
Generate metrics for one identity:
```bash
python -m evaluation.tests.classification_evaluation PersonName
```
Generate metrics for all identities:
```bash
python -m evaluation.tests.classification_evaluation --all
```
Script generates ROC and PR curves. ROC gives us insight into how the TPR (True positive rate) and FPR (False positive rate) change across different thresholds. Script also calculates the AUC-ROC (Area Under Curve), EER (Equal Error Rate) and AP (Average Precision). **EER** was used as a reference point to define all the **per-person thresholds**.

### 5. Performing PCA analysis
Perform PCA on the test-image embeddings:
```bash
python -m src.analysis.pca --mode faces
```
Perform PCA on the identity centroids:
```bash
python -m src.analysis.pca --mode centroids
```
Perform both analyses and save the resulting figures:
```bash
python -m src.analysis.pca --mode all --save
```
Showing multi-dimensional data on a 2D graph requires determining which components have the highest variance and applying the data onto those principal components. PCA analysis determines these components and displays all test-image and centroid embeddings on a 2D graph.
### 6. Running the recognition pipeline
Run temporal analysis on the test videos:
```bash
python -m pipeline --test
```
Run live face recognition using the default camera:
```bash
python -m pipeline 0
```
The argument 0 selects the first connected camera. Other camera indices can be supplied when multiple cameras are available.

Before running the pipeline please check:
- The centroids have been created
- Desired backend has been selected and thresholds have been defined in the ```config.yaml```
- Test videos or camera is accessible

## 🎯Results
Temporal analysis parameters:
- No. of videos of people with access - 7
- No. of videos of people without access - 3
- Temporal window - 5 embeddings
- Temporal stride - 5 frames
- Detection - MTCNN (factor=0.6)
- Preprocessing - Face alignment + Transforms w which the ViT Tiny was trained
- Embedding model - ViT Tiny quantized with ```quantize_(model, Int8WeightOnlyConfig())```
- Classification - Centroid-based 

Since the goal of this project was to evaluate and integrate Transformer models in an access control system, ViT Tiny model results will be presented here:
|Average AUC|Average EER|Average AP|Difference between avg intra and inter-class similarities|
|:-:|:-:|:-:|:-:|
|0.9171|0.1355|0.7278|0.1365|

Temporal analysis results for people with access:
| Class   |	FPS    |	Frame FRR % |	System FRR % |	Avg detection time (ms) |	Avg feature extraction time (ms) |
|---------|--------|----------------|----------------|--------------------------|----------------------------------|
| Antonio	| 119.20 | 25.00          | 0	             | 26.31                    | 19.03                            |
| David	  | 88.81	 | 13.80          | 0	             | 31	                      | 20.32                            |
| Ivor	  | 86.66	 | 14.87          | 0	             | 31.49                    | 21.00                            |
| Matej	  | 86.00	 | 29.03          | 16.67          | 30.57                    | 20.65                            |
| Mathias	| 77.39	 | 66.67          | 66.67          | 36.55                    | 27.10                            |
| Matija	| 95.96	 | 12.0	          | 20.0	         | 28.54                    | 20.51                            |
| Paolo	  | 103.39 | 4.76	          | 0	             | 26.33                    | 19.70                            |

Temporal analysis results for people without access:

| Class     |	FPS   |	Frame FRR % |	System FAR % |	Avg detection time (ms) |	Avg feature extraction time (ms) |
|-----------|------ |---------------|----------------|--------------------------|----------------------------------|
| Unknown 1	| 46.30	| 0	            | 0	             | 55.89	                  | 34.19                            |
| Unknown 2	| 60.02	| 0	            | 0	             | 56.77	                  | 23.00                            |
| Unknown 3	| 59.50	| 4.65	        | 0	             | 48.58	                  | 30.98                            |

These results show that the model has a **high FRR** for specific persons on a system level which can be due to the environment in which the test videos were taken and the possibility that that specific person has worse video and picture quality than the rest. **0% FAR** indicates that the model accurately refuses access to persons without access.

Limitations of these results is the amount of videos used for known persons and the small amount of test videos for people without access. 

Future work requires further optimization such as advanced quantization techniques, more videos in different environments.
