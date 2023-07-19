# GDSC6 - from Gandalf the Data-Wise and Aragorn the Code-Master

### Capgemini Global Data Science Challenge (GDSC) for a sustainable future!

![](https://gdsc.ce.capgemini.com/static/main_banner-13308435f9f145ca94c843a0a8fc4869.png)

GDSC6 invites participants to create a data and AI model for recognizing specific sounds, aiding in the identification and monitoring of insect species in remote habitats. The model aims to improve and expedite the identification process using cost-effective sensors that work day and night. Participants have the opportunity to make a positive impact while enhancing their AI skills and gaining certifications and targeted tutorials related to AI and AWS. #AI4Good is the driving force behind this challenge.


# Our solution
###### This repository contains the solution by the Team Gandalf the Data-Wise and Aragorn the Code-Master

**Our solution is built upon four key pillars, each playing a crucial role in its effectiveness:**

**1) Data Chunking:** We create data chunks with random lengths, optimizing the processing of information.

**2) Balanced Dataset:** A balanced dataset is generated from these chunks in an efficient manner, ensuring fair representation of different data points.

**3) Predictive Analysis:** For each audio, we conduct predictions by dividing it into maximum possible chunks of 11 seconds with 2-second steps. The final prediction is determined by selecting the class with the highest value.

**4) Simpler Model Training:** To prevent overfitting, we train on a simpler model, striking the right balance between complexity and generalization. Our model has 8 layers out of possible 12.

### Getting started
____________________
```
├── notebooks/
│   ├── Biodiversity_Buzz_Detection_Model_AWS_placeholder.ipynb
│   └── Biodiversity_Buzz_Detection_Model_placeholder.ipynb
├── src/
│   ├── auto_train.py
│   ├── config.py
│   ├── eda_utils.py 
│   ├── gdsc_eval.py
│   ├── preprocessing.py 
│   └── gdsc_utils.py 
├── .gitignore
└── README.md
```
#### Required libraries
```python
pip install datasets
pip install transformers
pip install transformers[torch]
pip install accelerate -U
```
