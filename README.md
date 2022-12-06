### Sturcture of the repository

The repository is structured as follows:

    .
    ├── data
    │   ├── X.npy
    │   ├── Y.npy
    │   └── Y_s.npy
    │
    ├── models
    │   └── Contains the trained models
    │
    ├── results
    │   └──  Contains the train, val loss and Confusion matrix 
    │   
    ├── tb_logs
    │   └── Contains the tensorboard logs
    │
    ├── Final.ipynb
    ├── model.py (Contains the model architecture and Feature extractor)
    ├── data.py (Contains the base class for the dataset and the datamodule)
    │
    ├── README.md
    └── requirements.txt (Contains the required packages)