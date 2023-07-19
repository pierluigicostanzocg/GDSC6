"""
This is the train script.

This script contains all steps required to train a Huggingface model.
"""

import logging                                                    # module for displaying relevant information in the logs
import sys                                                        # to access to some variables used or maintained by the interpreter 
import argparse                                                   # to parse arguments from passed in the hyperparameters
import os                                                         # to manage environmental variables
import json                                                       # to open the json file with labels
import math
from transformers import (                                        # required classes to perform the model training and implement early stopping
    AutoFeatureExtractor, 
    AutoModelForAudioClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    AutoConfig,
    set_seed
)
import torch, torchaudio                                           # library to work with PyTorch tensors and to figure out if we have a GPU available
from datasets import load_dataset, Audio, Dataset                  # required tools to create, load and process our audio dataset
import pandas as pd                                                # home of the DataFrame construct, _the_ most important object for Data Science
import numpy as np
from preprocessing import preprocess_audio_arrays                  # functions to preprocess the dataset with AutoFeatureExtractor
from gdsc_eval import compute_metrics, make_predictions            # functions to create predictions and evaluate them
from typing import List, Dict, Union, Any, Optional                # for type hints
import random

def get_feature_extractor(model_name: str, 
                          train_dataset_mean: Optional[float] = None, 
                          train_dataset_std: Optional[float] = None,
                          sr: int = 44100) -> AutoFeatureExtractor:
    """
    Retrieves a feature extractor for audio signal processing.

    Args:
        model_name (str): The name of the pre-trained model to use.
        train_dataset_mean (float, optional): The mean value of the training dataset. Defaults to None.
        train_dataset_std (float, optional): The standard deviation of the training dataset. Defaults to None.

    Returns:
        AutoFeatureExtractor: An instance of the AutoFeatureExtractor class.

    """
    if all((train_dataset_mean, train_dataset_std)):
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, mean=train_dataset_mean, std=train_dataset_std, sampling_rate=sr)
        logger.info(f" feature extractor loaded with dataset mean: {train_dataset_mean} and standard deviation: {train_dataset_std}")
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, sampling_rate=sr)
        logger.info(" at least one of the optional arguments (mean, std) is missing")
        logger.info(f" feature extractor loaded with default dataset mean: {feature_extractor.mean} and standard deviation: {feature_extractor.std}")
        
    return feature_extractor


def preprocess_function(examples: Dict[str, Any], path: bool = 1) -> Dict[str, Any]:
    """
    Preprocesses audio data for audio classification task.

    Parameters:
    -----------
        examples: dict
                  A dictionary containing the input examples, where the 'audio' key corresponds to the audio data.
                  Each audio example should have a 'path' and 'array' field.
        path: int (optional)
                   An integer flag indicating whether to include the 'file_name' field in the output.
                   Default is 1, which includes the 'file_name' field. Set to 0 to exclude it.

    Returns:
    --------
        dict: A dictionary containing the preprocessed inputs for audio classification.
              The returned dictionary includes the following fields:
              - 'input_values': The audio arrays preprocessed by the feature extractor, truncated to MAX_DURATION seconds.
              - 'label' (optional): The true labels of audio arrays.
              - 'attention_mask' (optional): If 'return_attention_mask' is True in the feature extractor, this field will be present.
              - 'file_name' (optional): If 'path' is set to 1, this field contains the filenames extracted from the 'path' field of input examples.
    """

    # Extract audio arrays from the input examples and truncate them to MAX_DURATION seconds.
    audio_arrays = [x["array"][:MODEL_SAMPLING_RATE*MAX_DURATION] for x in examples['audio']]

    # Use the feature extractor to preprocess the audio data.
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        return_attention_mask=False,
    )

    # Include 'file_name' field in the output if 'path' is set to 1.
    if path:
        inputs['file_name'] = [e['path'].split('/')[-1] for e in examples['audio']]

    return inputs


def chunk_aug(example: np.ndarray) -> np.ndarray:

    """
    Randomly selects a chunk from the input audio signal and returns the chunk with a maximum duration of MAX_DURATION seconds.

    Parameters:
    -----------
    example: numpy.ndarray
          The input audio signal represented as a 1-D numpy array.

    Returns:
    --------
        numpy.ndarray: A 1-D numpy array representing the selected chunk from the input audio signal 'example'. The maximum duration of the chunk is MAX_DURATION seconds, but it may be shorter if 'example' is not long enough.

    """

    if not isinstance(example, np.ndarray) or len(example) == 0:
      raise ValueError("Input 'example' must be a non-empty numpy.ndarray.")

    e_len = int(example.shape[0]/MODEL_SAMPLING_RATE) #length of audio
    min_len = min([2, e_len]) #min possible seconds
    max_len = min([MAX_DURATION, e_len]) #max possible seconds
    chunk_len = list(range(min_len, max_len+1)) #how many seconds
    chunk_len_ran = random.choice(chunk_len) #random chunk seconds

    e_len_range = list(range(0,e_len-chunk_len_ran+1)) #positions
    e_len_range_ran = random.choice(e_len_range) #random position
    example = example[e_len_range_ran*MODEL_SAMPLING_RATE:(e_len_range_ran+chunk_len_ran)*MODEL_SAMPLING_RATE] #get random chunk from audio

    return example[:MODEL_SAMPLING_RATE*MAX_DURATION]


def call_files(x: Dict[str, Union[List[int], torch.Tensor]]) -> Dict[str, Any]:

    """
    Loads audio files based on the labels in 'x', applies chunk augmentation, extracts features using a feature extractor,
    and returns the processed inputs.

    Parameters:
    -----------
    x : dict
        A dictionary containing the input data.
        Required keys:
            - 'label': A list of integers representing the labels.

    Returns:
    --------
    dict
        A dictionary containing the processed inputs.
        The dictionary has the following keys:
            - 'input_values': A torch.Tensor representing the processed audio inputs.
            - 'label': A list of integers representing the labels.
    """
    # Select a random file path for each label in 'x' from 'balanced_df_list'.
    path_files = [random.choice(balanced_df_list[l]) for l in x['label']]

    # Extract the file names from the selected file paths.
    file_name = [p.split('/')[-1] for p in path_files]

    # Apply chunk augmentation to the audio signals and store the augmented chunks in 'wv'.
    wv = [chunk_aug(np.array(torchaudio.load(p)[0][0].numpy())) for p in path_files]

    # Extract features from the augmented audio signals using the feature extractor.
    inputs = feature_extractor(
        wv,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        return_attention_mask=False)

    # Convert the 'input_values' to a torch.Tensor and store it in the 'inputs' dictionary.
    inputs['input_values'] = torch.Tensor(np.array(inputs['input_values']))

    # Store the 'label' values from the input dictionary 'x' in the 'inputs' dictionary.
    inputs['label'] = x['label']

    return inputs

def chunk_pred(example: np.ndarray) -> List[np.ndarray]:
    """
    Divide the input audio signal 'example' into multiple chunks and return a list of these chunks for prediction.

    Parameters:
    -----------
    example : numpy.ndarray
        The input audio signal represented as a 1-D numpy array.

    Returns:
    --------
    List[numpy.ndarray]
        A list of numpy arrays, each representing a chunk of the input audio signal for prediction.

    """
    e_len = int(example.shape[0]/MODEL_SAMPLING_RATE)
    if e_len > MAX_DURATION: #if length of audio is more than MAX_DURATION seconds, divide the audio to chunks
        min_len = min(360, e_len-11) #min possible seconds
        chunk_len = list(range(0, min_len, 2)) #how many seconds
        return [e[MODEL_SAMPLING_RATE*r:MODEL_SAMPLING_RATE*(MAX_DURATION+r)] for r in chunk_len]
    return [e[:MODEL_SAMPLING_RATE*MAX_DURATION]]

def preprocess_function_pred_chunks(examples: Dict[str, Any], model: torch.nn.Module) -> Dict[str, Any]:

    """
    Preprocesses audio examples in chunks for prediction using the provided model.

    Parameters:
    -----------
    examples : dict
        A dictionary containing the audio examples.
        The 'audio' key holds another dictionary with two keys: 'array' (numpy.ndarray) and 'path' (str).
    model : torch.nn.Module
        The audio classification model used for prediction.

    Returns:
    --------
    dict
        A dictionary containing the processed audio examples with additional prediction information.
        The keys in the dictionary include:
            - 'audio': A dictionary with the processed audio data.
            - 'file_name': A string representing the file name of the audio example.
            - 'pred_id': A list of tuples containing the class ID and its corresponding prediction score for each chunk.
            - 'predicted_class_id': An integer representing the predicted class ID for the example.
    """

    # Chunk the audio data using the 'chunk_pred' function
    wv = chunk_pred(examples['audio']['array'])

    # Extract features from the chunked audio data using the feature extractor
    inputs = feature_extractor(
        wv,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        return_attention_mask=False,
    )

    # Move the model to the 'cuda:0' device for GPU acceleration
    model_pred = model.to('cuda:0')

    # Convert the input values to a torch.Tensor and move it to the 'cuda:0' device
    input_values = torch.Tensor(np.array(inputs['input_values'])).to('cuda:0')

    # Perform predictions for each chunk using the model and store the logits
    logits = []
    with torch.no_grad():
        logits = [model(i.unsqueeze(0)).logits.cpu() for i in input_values]

    # Concatenate the logits and extract the predicted class IDs and their corresponding prediction scores
    logits = torch.Tensor(np.concatenate(logits))
    predicted_class_id_max = [torch.max(item).item() for item in logits]
    predicted_class_id = [(int(torch.argmax(item).item()), torch.max(item).item()) for item in logits]

    # Find the class ID with the highest prediction score for the entire example
    pred_id_max = torch.Tensor(predicted_class_id)[:, 1:].argmax().item()

    # Update the 'examples' dictionary with prediction information and file name
    examples['pred_id'] = predicted_class_id
    examples['predicted_class_id'] = predicted_class_id[pred_id_max][0]
    examples['file_name'] = examples['audio']['path'].split('/')[-1]

    # Return the updated 'examples' dictionary with prediction information
    return examples

if __name__ == "__main__":
    random.seed(42)                                                                        # for model reproducibility
    set_seed(42)                                                                           # for model reproducibility

    parser = argparse.ArgumentParser()
    if 'SM_MODEL_DIR' not in os.environ: os.environ['SM_MODEL_DIR'] = ''                   # for compatibility out of sagemaker
    if 'SM_CHANNEL_DATA' not in os.environ: os.environ['SM_CHANNEL_DATA'] = ''             # for compatibility out of sagemaker

    # hyperparameters sent from our jupyter notebook are passed as command-line arguments to the script
    # preprocessing hyperparameters
    parser.add_argument("--sampling_rate", type=int, default=44100)                        # sampling rate to which we will cast audio files
    parser.add_argument("--fe_batch_size", type=int, default=32)                           # feature extractor batch size
    parser.add_argument("--train_dataset_mean", type=float, default=None)                  # mean value of spectrograms of our data
    parser.add_argument("--train_dataset_std", type=float, default=None)                   # standard deviation value of spectrograms of our resampled data
    
    # training hyperparameters
    parser.add_argument("--model_name", type=str)                                          # name of the pretrained model from HuggingFace
    parser.add_argument("--learning_rate", type=float, default=5e-5)                       # learning rate
    parser.add_argument("--epochs", type=int, default=1)                                   # number of training epochs 
    parser.add_argument("--train_batch_size", type=int, default=4)                         # training batch size
    parser.add_argument("--eval_batch_size", type=int, default=4)                          # evaluation batch size
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)              # the number of gradient accumulation steps to be used during training.
    parser.add_argument("--num_hidden_layers", type=int, default=8)                        # number of hidden layers to prune the model
    parser.add_argument("--patience", type=int, default=2)                                 # early stopping - how many epoch without improvement will stop the training 
    parser.add_argument("--data_channel", type=str, default=os.environ["SM_CHANNEL_DATA"]) # directory where input data from S3 is stored
    parser.add_argument("--train_dir", type=str, default="train")                          # folder name with training data
    parser.add_argument("--val_dir", type=str, default="val")                              # folder name with validation data
    parser.add_argument("--test_dir", type=str, default="test")                            # folder name with test data
    parser.add_argument("--output_dir", type=str, default=os.environ['SM_MODEL_DIR'])      # output directory. This directory will be saved in the S3 bucket
    
     
    args, _ = parser.parse_known_args()                    # parsing arguments from the notebook

    MAX_DURATION = 11                                                                      # max duration in  of the audio files (generally to pass to the feature extractor - it speed up the processing by 10x)
    MODEL_SAMPLING_RATE = args.sampling_rate
    MY_PATH = '/'.join((args.data_channel).split('/')[:-1])                                # align path with the list of path from the metadata file 
    
    train_path = f"{args.data_channel}/{args.train_dir}"   # directory of our training dataset on the instance
    val_path = f"{args.data_channel}/{args.val_dir}"       # directory of our validation dataset on the instance
    test_path = f"{args.data_channel}/{args.test_dir}"     # directory of our test dataset on the instance

    
    # Set up logging which allows to print information in logs
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Torch version")
    logger.info(torch.__version__)
    logger.info("Torch sees CUDA?")
    logger.info(torch.cuda.is_available())
    
    # Load json file with label2id mapping
    with open(f'{args.data_channel}/labels.json', 'r') as f:
        labels = json.load(f)
    
    # Create mapping from label to id and id to label
    label2id, id2label = dict(), dict()
    for k, v in labels.items():
        label2id[k] = str(v)
        id2label[str(v)] = k
    
    num_labels = len(label2id)  # define number of labels

    
    # If mean or std are not passed it will load Featue Extractor with the default settings.
    feature_extractor = get_feature_extractor(args.model_name, sr=args.sampling_rate)

    # creating validation and test datasets
    val_dataset = load_dataset("audiofolder", data_dir=val_path).get('train')
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=MODEL_SAMPLING_RATE))
    val_dataset_encoded = val_dataset.map(
        lambda x: preprocess_function(x, path=0), remove_columns=["audio"], batched=True, batch_size=2)
    val_dataset_encoded.set_format(type='torch', columns=['input_values'], output_all_columns=True)
    
    test_dataset = load_dataset("audiofolder", data_dir=test_path).get('train')
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=MODEL_SAMPLING_RATE))
    test_dataset_encoded = test_dataset.map(
        preprocess_function, remove_columns=["audio"], batched=True, batch_size=2)
    test_dataset_encoded.set_format(type='torch', columns=['input_values'], output_all_columns=True)

    logger.info("Val and test ready")


    # creating train datasets
    balanced_dataset_dict = {
        'audio': ['']*20*66,
        'label': list(range(66))*20,
    }
    balanced_dataset = Dataset.from_dict(balanced_dataset_dict)
    all_data = pd.read_csv(f'{args.data_channel}/metadata.csv')
    all_data['path'] = f'{MY_PATH}/' + all_data['path']
    balanced_df_list = all_data[['label','path']].groupby('label')['path'].apply(list)
    balanced_dataset.set_transform(call_files, output_all_columns=True)

    # Download model from model hub
    model_config = AutoConfig.from_pretrained(args.model_name, num_labels=num_labels, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True, num_hidden_layers=args.num_hidden_layers)

    # Create an audio classification model 'model' using the AutoModelForAudioClassification class.
    model = AutoModelForAudioClassification.from_pretrained(args.model_name, config=model_config, ignore_mismatched_sizes=True)
    
    # Define training arguments for the purpose of training
    training_args = TrainingArguments(
        output_dir=args.output_dir,                          # directory for saving model checkpoints and logs
        num_train_epochs=args.epochs,                        # number of epochs
        per_device_train_batch_size=args.train_batch_size,   # number of examples in batch for training
        per_device_eval_batch_size=args.eval_batch_size,     # number of examples in batch for evaluation
        evaluation_strategy="epoch",                         # makes evaluation at the end of each epoch
        learning_rate=args.learning_rate,                    # learning rate
        optim="adamw_torch",                                 # optimizer
        # warmup_ratio=0.1,                                  # warm up to allow the optimizer to collect the statistics of gradients
        logging_steps=1,                                     # number of steps for logging the training process - one step is one batch; float denotes ratio of the global training steps
        load_best_model_at_end = True,                       # whether to load or not the best model at the end of the training
        metric_for_best_model="eval_loss",                   # claiming that the best model is the one with the lowest loss on the val set
        save_strategy = 'epoch',                             # saving is done at the end of each epoch
        disable_tqdm=True,                                   # disable printing progress bar to reduce amount of logs
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,                                                                 # passing our model
        args=training_args,                                                          # passing the above created arguments
        compute_metrics=compute_metrics,                                             # passing the compute_metrics function that we imported from gdsc_eval module
        train_dataset=balanced_dataset,                                              # passing the balanced train set
        eval_dataset=val_dataset_encoded,                                            # passing the encoded val set
        tokenizer=feature_extractor,                                                 # passing the feature extractor
        callbacks = [EarlyStoppingCallback(early_stopping_patience = args.patience)] # adding early stopping to avoid overfitting
    )

    # Train the model
    logger.info(f" starting training proccess for {args.epochs} epoch(s)")  
    trainer.train()
 
    # Prepare predictions on the validation set for the purpose of error analysis
    logger.info(" training job done. Preparing predictions for validation set.")
     
    # use gpu for inference if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Preparing predictions for test set and saving them in the output directory
    test_dataset_encoded_pred_chunks = test_dataset.map(lambda x: preprocess_function_pred_chunks(x, model),
                                                              remove_columns=["audio"],
                                                              batched=False,
                                                              batch_size=1)
    test_dataset_encoded = test_dataset_encoded.map(lambda x: make_predictions(x['input_values'], model, device), batched = True, batch_size=args.eval_batch_size)   
    
    # Keeping only the important columns for the csv file
    test_dataset_encoded_pred_chunks_pandas_df = test_dataset_encoded_pred_chunks.to_pandas()[['file_name', 'predicted_class_id']]
    test_dataset_encoded_pred_chunks_pandas_df.to_csv(f"{args.output_dir}/prediction_test.csv", index = False)  # saving the file with test predictions
    
    logger.info(" prepared predictions for test set and saved it to the output directory. Training job completed")
