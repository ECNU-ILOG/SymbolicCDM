# SymbolicCDM

This repository contains the code for the paper "Symbolic Cognitive Diagnosis via Hybrid Optimization for Intelligent Education Systems" 
published in _proceedings of the 38th AAAI Conference on Artificial Intelligence_. 

Besides, we also upload the main paper and its appendix, titled `main.pdf` and `appendix.pdf` respectively in the folder named paper.

![Framework](asset/img.png)

**Should you have any questions, add issues in this repository. We will try our best to address your concerns.**

## Requirements
Create the running environment with conda `4.10.3` with Python `3.9.0`:
```
conda create -n scdm python==3.9
conda activate scdm
```

Install all necessary packages:
```
pip install -r requirements.txt
```

## Reproducing
### Run `example.py`
We have prepared a sample dataset FracSub to demonstrate the SCD framework. Having installed
all necessary packages, you can run `example.py` using
```
python example.py
```

### Run with other datasets
#### Step 1. prepare dataset
Refer to the sample dataset, you should prepare the following files:
```
├─dataset
│  └─Your_dataset
│          config.json
│          data.csv   
│          q.csv
```
Specifically, `config.json` records all necessary settings of dataset like the number of students, and the format of config.json is shown as following:
```
{
  "dataset": [String, the name of the dataset],
  "qMatrixPath": [string, the relative path of Q matrix],
  "dataPath": [string, the relative path of response logs],
  "studentNumber": [int, the number of students],
  "questionNumber": [int, the number of questions],
  "knowledgeNumber": [int, the number of knowledge attributes]
}
```

`data.csv` consists of response logs in the following format:
```
[int, student_id1],[int, question_id1],[0/1, response to question_id1]
[int, student_id1],[int, question_id2],[0/1, response to question_id2]
...
[int, student_idn],[int, question_idm],[0/1, response to question_idm]
```

`q.csv` contains the relevant between questions and knowledge attributes. Each entry in the $i$-th row and the $j$-th column means
whether the $i$-th question involves the $j$-th knowledge attributes.

#### Step 2. coding
Refer to the `example.py`, you can change the path to different configuration file.

#### Step 3. run code
```
python example.py
```



## File Tree
```
SymbolicCDM:
│  example.py
│  LICENSE
│  README.md
│  requirements.txt   
│      
├─asset
│      img.png        
│      
├─dataset
│  └─FracSub
│          config.json
│          data.csv   
│          q.csv
│
├─paper
│      appendix.pdf
│      main.pdf
│
└─SCDM
      eval.py
      interaction.py
      model.py
      operators.py
      parameter.py
      utility.py
      __init__.py

```

## Citation

```
@inproceedings{shen2024symbolic,
 author = {Shen, Junhao and 
           Qian, Hong and 
           Zhang, Wei and 
           Zhou, Aimin},
 booktitle = {Proceedings of the 38th AAAI Conference on Artificial Intelligence},
 title = {Symbolic Cognitive Diagnosis via Hybrid Optimization for Intelligent Education Systems},
 year = {2024},
 address = {Vancouver, Canada},
 page = {}
}
```
