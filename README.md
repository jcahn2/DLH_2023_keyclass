# DLH_2023_keyclass
Final project for Deep Learning for Healthcare 598 class
## KeyClass Paper
* [KeyClass_Classifying_Unstructured_Clinical_Notes_with_Automatic_Weak_Supervision.pdf](https://arxiv.org/pdf/2206.12088.pdf)

## KeyClass Github
* https://github.com/autonlab/KeyClass

## Dependencies
* Snorkel
* SentenceTransformers
* Pytorch
* Pandas
* Numpy

## Data Download
- Download NOTEEVENTS.csv and DIAGNOSES_ICD.csv from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/)
- Set absolute path to the above CSVs [here](https://github.com/jcahn2/DLH_2023_keyclass/blob/35eb003ffff1b98158a20d422bf20f11f8c41750/KeyClass/createAdmissionNoteTable.R#L25) and [here](https://github.com/jcahn2/DLH_2023_keyclass/blob/35eb003ffff1b98158a20d422bf20f11f8c41750/KeyClass/createAdmissionNoteTable.R#L30)
- Run [createAdmissionNoteTable.R](https://github.com/jcahn2/DLH_2023_keyclass/blob/main/KeyClass/createAdmissionNoteTable.R)
- Place the output files (icd9NotesDataTable_train.csv, icd9NotesDataTable_valid.csv) in KeyClas/fastag_data
- Make the directory KeyClass/data/mimic/

## Run 
* Run [keyclass_mimic.py](https://github.com/jcahn2/DLH_2023_keyclass/blob/main/KeyClass/experiments/keyclass_mimic.py) from the terminal (from the KeyClass/experiments folder)

[keyclass_mimic.py](https://github.com/jcahn2/DLH_2023_keyclass/blob/main/KeyClass/experiments/keyclass_mimic.py) contains all information on the preprocessing, training, pretrained models, and evaluaion code specified in the requirements. Each section is commented and highlighted.

## Results

### Comparison to Original KeyClass and FasTag
| Score     | KeyClass with Fully Connected | Original KeyClass | FasTag |
|-----------|-------------------------------|-------------------|--------|
| Precision | 0.346                         | 0.507             | 0.436  |
| Recall    | 0.244                         | 0.896             | 0.734  |
| F1        | 0.286                         | 0.625             | 0.525  |

### BlueBERT vs MPNET

| Score     | BlueBERT | MPNET |
|-----------|----------|-------|
| Precision | 0.346    | 0.453 |
| Recall    | 0.244    | 0.471 |
| F1        | 0.286    | 0.462 |

### TCN vs Fully Connected 

| Score     | BERT FC | MPNET FC | BERT TCN | MPNET TCN |
|-----------|---------|----------|----------|-----------|
| Precision | 0.346   | 0.453    | 0.115    | 0.074     |
| Recall    | 0.244   | 0.471    | 0.024    | 0.034     |
| F1        | 0.286   | 0.462    | 0.039    | 0.046     |
