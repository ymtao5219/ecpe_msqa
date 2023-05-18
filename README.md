# Emotion cause pair extraction as a question answering problem

The repository contains code for CS685 NLP Project in Spring 2023 at Umass Amherst. 

## Dependencies: 

- Python 3.7 
- transformers
- pytorch
- Pretrained BERT "chinese-bert-base", which is available from huggingface: https://huggingface.co/bert-base-chinese 

We have three tasks to complete in our project proposal. We orgainize our code according to the tasks.  

## Task 0: Statistical analysis of the dataset
- 'task0/distribution.ipynb': contains the statistical analysis of the dataset. 


## Task 1: Our proposed method
- 'task1/ECPE-MLL': contains the code of the paper. Their code is publicly availabel at: https://github.com/NUSTM/ECPE-MLL. The authors use Tensorflow 1.x and Python 2. 
- `task1/ECPE-MLL-Pytorch`: contains our proposed method to solve this task 
- The below are commands to run the code:
```bash
cd task1/ECPE-MLL-Pytorch
# The hyperparameters are stored in the config.py file. 
# They may also be modified in a dictionary named "mod_para" in train.py for grid search.
python3 -u train.py 2>&1 | grep -E ">>>>|===="
```
- Compute resources: 
    - We utilize Google Cloud Compute services for model training. The hardware consists of an Intel Skylake CPU (13GB) platform along with a single NVIDIA T4 GPU (with 16GB of memory). 
    - In order to execute our code successfully, it is necessary for the GPU to possess a memory capacity exceeding 8GB.
    

## Task 2: Evaluting the ECPE task using GPT3.5 
- 'task2/zeroshot_example': contains the code for an zeroshot learning example with ChatGPT. Running the codes will save the response of ChatGPT to the text. You may need to use your own API keys to call ChatGPT.
- The below are commands to run the code:
```
python zeroshot.py --api_keys <your API keys>
```



## Task 3:ECPE as an extractive question-answering problem 
- `task3/`: contain code for our experiments. We tried conditional random field model with a combination of BiLSTM. Unfortunately, we were unable to complete this task due to the significant challenges encountered in Task 1, which took longer than expected.
