# multiattempt-bci
This is an implementation for validating a multiple attempt approach in Brain-Computer Interface (BCI).
###The application is used to:
 - measure EEG signals from Emotiv EEG headset;
 - preprocess EEG signal data using Short-Time Fourier Tranform;
 - store the preprocessed data;
 - run training and validation of a machine learning algorithm (Random Forest);
 - apply ensemble techniques on classified data, to get different set of prediction accuracies;
 - calculate the required number of concentration attempts required to reach the accuracy of 99%;
 - provide measurement of EEG signals considering the number of attempts required to reach 99%.

The idea of multiple attempt approach is briefly showed on the following picture, where the length of one concentration is 10 seconds:

![Alt text](/thesis/figures/main.png?raw=true "Multiple attempt approach compared to a single attempt")

The given project was done in scope of Master's thesis for the University of Tartu. 
