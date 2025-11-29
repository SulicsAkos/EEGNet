# EEGNet
A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces

# Introduction
A Collection of Convolutional Neural Network (CNN) models for EEG signal processing and classification, written in Keras and Tensorflow. The aim of this project is to
- provide a set of well-validated CNN models for EEG signal processing and classification
- facilitate reproducible research and
- enable other researchers to use and compare these models as easy as possible on their data

# Requirements

- Python == 3.7 or 3.8
- tensorflow == 2.X (verified working with 2.0 - 2.3, both for CPU and GPU)
To run the EEG/MEG ERP classification sample script, you will also need

- mne >= 0.17.1
- PyRiemann >= 0.2.5
- scikit-learn >= 0.20.1
- matplotlib >= 2.2.3

# Models Implemented

- EEGNet
- DeepConvNet
- ShallowConvNet

# Usage

To use this package, place the contents of this folder in your PYTHONPATH environment variable. Then, one can simply import any model and configure it as

from EEGModels import EEGNet, ShallowConvNet, DeepConvNet
model  = EEGNet(nb_classes = ..., Chans = ..., Samples = ...)
model2 = ShallowConvNet(nb_classes = ..., Chans = ..., Samples = ...)
model3 = DeepConvNet(nb_classes = ..., Chans = ..., Samples = ...)

Compile the model with the associated loss function and optimizer (in our case, the categorical cross-entropy and Adam optimizer, respectively). Then fit the model and predict on new test data.

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
fittedModel    = model.fit(...)
predicted      = model.predict(...)

