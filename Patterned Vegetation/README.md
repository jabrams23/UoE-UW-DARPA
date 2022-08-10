Code to train a DL model to identify patterned vegetation. Current



GeneratePattern - contains an implementation of code to generate patterned vegetation with varying levels of precipitation from a model developed by Konings et al. (2011): https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011JG001748

CreatePatternDatasets - generate vegetation patterns and non-patterns for training dataset. Currently set up for patterns of 150x150 pixels.

TrainDLModel - reshape pattern datasets and train DL model with them.
