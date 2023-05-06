# DLH_2023_keyclass
Final project for Deep Learning for Healthcare 598 class
## KeyClass Paper
* https://static1.squarespace.com/static/59d5ac1780bd5ef9c396eda6/t/62e979ca698f156547773868/1659468235402/114+MLHC_2022__KeyClass_Classifying_Unstructured_Clinical_Notes_with_Automatic_Weak_Supervision.pdf

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
