This repository contains a report that presents our approach for the ”Influencer or Observer” social role classification challenge and two notebooks that allow to get the .csv for the kaggle submission.

The notebook containing our best pipeline, using XGBoost, is called Kaggle_code_Intweetionists.ipynb.

To run this notebook, the following two data files must be placed in the same repository folder: 
- kaggle_test.jsonl 
- train.jsonl

The entire process of our best pipeline, including training and inference, is expected to complete in approximately 10 minutes on a standard computer.

The second notebook called DL_Intweetionists.ipynb contains a second pipeline, fully based on deep learning models to classify the tweets. Its accuracy is for now a bit lower than the best one obtained with XGBoost which is 0.842.