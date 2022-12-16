# Deep-StRIP
Deep Learning Approach for Structural Repeat Identification in Proteins

## Getting Started

Install the requirements to setup the environment 
```
pip install -r requirements.txt
```
### Protein Repeat type Classification
The input to the model is specified in ```input_data_repeat_classification``` folder. Extra input samples can be added from the  ```protein_DMs_samples``` folder to the ```input_data_repeat_classification``` folder for classification purpose. The model loads a pre-trained model from ```saved_models``` folder and outputs predicted repeat type as Class III, Class IV or Non-repeat.

```
python3 protein_repeat_classification.py
```

### Residue Classification
The input to the model is specified in ```input_data_residue_classification``` folder. Extra input samples can be added from the  ```residue_data_samples``` folder to the ```input_data_residue_classification``` folder for classification purpose. The model loads a pre-trained model from ```saved_models``` folder and outputsthe predicted repeat region. The predicted output is also stored in the ```results``` folder.

```
python3 residue_classification.py
```

## Introduction
Deep-StRIP is a Deep Learning Approach for Structural Repeat Identification in Proteins. The method uses the structural information encoded in the internal distances of the distance matrix to train a Convolutional Neural Network to predict the Repeat Type (Class III/ Class IV/ Non-repeat) and their Repeat Region.
