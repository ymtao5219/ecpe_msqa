# Emotion cause pair extraction as a question answering problem

CS685 NLP Project in Spring 2023 at Umass 

We have three tasks to complete in our project proposal: 

## Task 1: Replication of one of SOTA works
- 'task1/ECPE-MLL': contains the code of the paper. Their code is publicly availabel at: https://github.com/NUSTM/ECPE-MLL. The authors use Tensorflow 1.x and Python 2. 
- `task1/ECPE-MLL-Pytorch`: contains our replication of the above work. 
  - The below are commands to run the code:
  ```bash
  cd task1/ECPE-MLL-Pytorch
  # the hyperparameters are stored in the config.py file
  python train.py 
  ```
  - Compute resources: 
    - We utilize Google Cloud Compute services for model training. The hardware consists of an Intel Skylake CPU platform along with a single NVIDIA T4 GPU (with 16GB of memory). 
    - In order to execute our code successfully, it is necessary for the GPU to possess a memory capacity exceeding 8GB.
    
    

## Task 2: Evaluting the ECPE task using GPT3.5 
- `task2/`: 




## Task 3:ECPE as an extractive question-answering problem 
- `task3/`: contain code for our experiments 
