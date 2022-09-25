# Setup
1. Clone the repo
```
git clone https://github.com/steviedale/dlc_2021.git
```

2. Create a virtual environment
```
python3 -m venv venv
```

3. Source virtual environment
```
source venv/bin/activate
```

4. Install pip packages
```
pip install -r requirements.txt
```

5. Rename absolute paths in .csv file.
    1. Go to dataset/dataframes/10k.csv
    2. find/replace "/home/stevie/git/dlc_2021" with the path to this repo

# Run Experiments

## KNN
1. run the jupyter notebook at knn/knn.ipynb 
2. create and view the plots by running the notebook at plotting/plot_knn.ipynb

## Decision Tree
1. run the jupyter notebook at decision_tree/decision_tree.ipynb 
2. create and view the plots by running the notebook at plotting/plot_decision_tree.ipynb

## Boosting
1. run the jupyter notebook at boosting/boosting.ipynb 
2. create and view the plots by running the notebook at plotting/plot_boosting.ipynb

## SVM
1. run the jupyter notebook at svm/svm.ipynb 
2. create and view the plots by running the notebook at plotting/plot_svm.ipynb

## Neural Networks
1. source the virtual environment
```
source venv/bin/activate
```
2. change directory to neural_network dir
```
cd neural_network
```
3. run training
```
python train.py
```
Note: I used weights and biases to log my learning curves. Unfortunately I can't give you access to my wandb. But you are welcome to
update the commented out code that refers to wandb and add in your own wandb info or another logger.