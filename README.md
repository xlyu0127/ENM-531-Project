# ENM-531-Project
This is the final project report for Upenn ENM 531
traindata, testdata are the raw data whose first column is the SDBS number of the compound, second column is the bond matrix, third column is the chemical shift (labeled to −1 if no hydrogen is attached). r4testcode,r4testY, r4traincode, r4trainY are test and training data of BNN with the −1 entries being removed. MGNNEMVreverse.py and bnn2.py are script files containing class and function definitions for GCNNAFP and BNN. encode result.py generates atomic fingerprint from raw data to atomic finger print, after the state dictionary r4e is loaded for GCNNAFP. Commented blocks in full4r generate most of the plots. 'dataaq' is an auxiliary file for data aquisition.
