# Deep-PRIDM
A Deep Learning method for Protein Repeat Identification using Distance Matrix analysis
![Deep-PRIDM_logo_2](https://user-images.githubusercontent.com/18621779/113302665-f1808b00-931d-11eb-89f5-4e180607f987.png)

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
The input to the model is specified in ```input_data_residue_classification``` folder. Extra input samples can be added from the  ```residue_data_samples``` folder to the ```input_data_residue_classification``` folder for classification purpose. The model loads a pre-trained model from ```saved_models``` folder and outputsthe predicted repeat region along with start/end boundaries of the repeating unit along with copy number. The predicted output is also stored in the ```results``` folder.

```
python3 residue_classification.py
```

## Introduction
Deep-PRIDM is Deep Learning method for Protein Repeat Identification using Distance Matrix analysis. The method uses the structural information encoded in the internal distances of the distance matrix to train a Convolutional Neural Network to predict the Repeat Type (Class III/ Class IV/ Non-repeat), Repeat Region with start/end boundaries and Copy number of a protein structure.
