# DeNN
A deep neural network framework for denosing functional MRI(fMRI) data

===============================================================
### Instruction

### 1. Purpose of DeNN
- DeNN is a non-regression based deep learning technique for de-noising fMRI data, which is applicable for both resting state and task fMRI data.

### 2. Preprocessing
* Purpose of preprocessing
This is a step to save a .mat file for each individual subject, which is the input data for DeNN denoising. Technically it should not be called as preprocessing. :)
* Required input files
4D fMRI data, structural MRI data, and segmented MRI images (gray matter, white matter and cerebrospinal fluid). These images should be in the same space, either the native space or the template space.
4D fMRI data should have been preprocessed with slice-timing correction, realignment. Any other preprocessing steps are optional.
* preprocessing script
We have shared the preprocessing script (ADNI_extractsegdata.m), which we have used for ADNI data. Researchers can change the directory to their own data
### 3. DeNN denoising
The scripts for DeNN is currently not publicly available. However, researchers can reach us by sending email to yzhengshi@gmail.com and provide us the .mat file from preprocessing script for DeNN denoising. Please keep the data anonymous.
### 4. Reference
Please cite the following reference if you use DeNN in your research.
* Yang et al., Disentangling time series between brain tissues improves fMRI data quality using a time-dependent deep neural network.

Other two references you may have interest:
Deep neural network for task fMRI denoising
* Yang et al., (2020) A robust deep neural network for denoising task-based fMRI data: An application to working memory and episodic memory. Medical Imaging Analysis, https://doi.org/10.1016/j.media.2019.101622.
Regression based deep neural network for resting state fMRI denoising
* Yang et al., (2019) Robust Motion Regression of Resting-State Data Using a Convolutional Neural Network Model. Front. Neurosci., 28 February 2019 | https://doi.org/10.3389/fnins.2019.00169.

