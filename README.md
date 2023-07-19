# GDSC6 - from Gandalf the Data-Wise and Aragorn the Code-Master

### Capgemini Global Data Science Challenge (GDSC) for a sustainable future!

![](https://gdsc.ce.capgemini.com/static/main_banner-13308435f9f145ca94c843a0a8fc4869.png)

GDSC6 invites participants to create a data and AI model for recognizing specific sounds, aiding in the identification and monitoring of insect species in remote habitats. The model aims to improve and expedite the identification process using cost-effective sensors that work day and night. Participants have the opportunity to make a positive impact while enhancing their AI skills and gaining certifications and targeted tutorials related to AI and AWS. #AI4Good is the driving force behind this challenge.


# Solution
### Our solution is built upon four key pillars, each playing a crucial role in its effectiveness:

**1) Data Augmentation Chunking:** We created a custom function for chunking the audio files with random lengths and position, to diversify and optimize the flow of information during training - as an augmentation technique.

**2) Balanced Dataset:** The training dataset has been balanced with a generator for ensuring fair representation of different audio instances for each classes.

**3) Inference:** At inference time, each audio divided into a maximum possible chunks of 11 seconds with 2-second time steps. The final prediction is determined by selecting the class with the highest score value.

**4) Simpler Model Training:** To prevent overfitting, we reduced the complexity of the AST model to a simpler one via model pruning layer technique. It ensure a better generalization. The standard AST model has 12 Multi-Head attention layers which we reduce it to 8.

### Getting started
____________________
```
├── notebooks/
│   ├── Biodiversity_Buzz_Detection_Model_AWS.ipynb
│   └── Biodiversity_Buzz_Detection_Model.ipynb
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
