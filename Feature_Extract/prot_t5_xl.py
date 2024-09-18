import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

# Model name to load the pre-trained protein language model
model_name = 'Rostlab/prot_t5_xl_uniref50'
# Load T5Tokenizer for converting sequence text to model input format
tokenizer = T5Tokenizer.from_pretrained(model_name)
# Load T5EncoderModel as the base model for feature extraction
model = T5EncoderModel.from_pretrained(model_name)

# Read training dataset
df = pd.read_csv('../Dataset/classification_train.csv')

# Get sequence data and preprocess it by separating each amino acid with spaces
sequences = df['Sequence'].tolist()
sequences = [" ".join(seq) for seq in sequences]

# Get the labels and reshape them into a 2D array
label = df['label'].tolist()
label = np.array(label).reshape(-1, 1)

# Define the maximum sequence length for truncation and padding
max_length = 17

# Use the tokenizer to encode the sequences, returning PyTorch tensors
encoded_inputs = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

# Forward pass through the model to get the representations (feature vectors) of the sequences
with torch.no_grad():
    outputs = model(**encoded_inputs)

# Get the last hidden state output from the model
sequence_output = outputs.last_hidden_state

# Get batch size, sequence length, and feature size
batch_size, seq_len, features = sequence_output.shape
# Transpose and reshape the output for pooling operation
sequence_output_reshaped = sequence_output.transpose(1, 2).reshape(batch_size * features, 1, seq_len)

# Perform 1D average pooling with a kernel size equal to the sequence length
kernel_size = 17
pooling_output = F.avg_pool1d(sequence_output_reshaped, kernel_size=kernel_size, stride=kernel_size)

# Reshape the pooled output and apply mean pooling over the features
pooled_output_reshaped = pooling_output.reshape(batch_size, features, -1).mean(dim=2)
# Concatenate the labels with the features into a single data matrix
data = np.hstack((label, pooled_output_reshaped.numpy()))

# Save the concatenated data to a CSV file
features_df = pd.DataFrame(data, columns=['label'] + [f'Feature_{i+1}' for i in range(pooled_output_reshaped.shape[1])])
output_file = '../Feature/classification/xl_train1.csv'
features_df.to_csv(output_file, index=False)
# Print save confirmation and the shape of the pooled features
print('已保存')
print(pooled_output_reshaped.shape)


##############################
# Code for processing the test set is identical to the training set
model_name = "Rostlab/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)

# Read test dataset
df = pd.read_csv('../Dataset/classification_test.csv')

# Get sequence data and preprocess it by separating each amino acid with spaces
sequences = df['Sequence'].tolist()
sequences = [" ".join(seq) for seq in sequences]
# Get the labels and reshape them into a 2D array
label = df['label'].tolist()
label = np.array(label).reshape(-1, 1)

# Define the maximum sequence length for truncation and padding
max_length = 17

# Use the tokenizer to encode the sequences, returning PyTorch tensors
encoded_inputs = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

# Forward pass through the model to get the representations (feature vectors) of the sequences
with torch.no_grad():
    outputs = model(**encoded_inputs)

# Get the last hidden state output from the model
sequence_output = outputs.last_hidden_state

# Get batch size, sequence length, and feature size
batch_size, seq_len, features = sequence_output.shape

# Transpose and reshape the output for pooling operation
sequence_output_reshaped = sequence_output.transpose(1, 2).reshape(batch_size * features, 1, seq_len)

# Perform 1D average pooling with a kernel size equal to the sequence length
kernel_size = 17
pooling_output = F.avg_pool1d(sequence_output_reshaped, kernel_size=kernel_size, stride=kernel_size)

# Reshape the pooled output and apply mean pooling over the features
pooled_output_reshaped = pooling_output.reshape(batch_size, features, -1).mean(dim=2)

# Concatenate the labels with the features into a single data matrix
data = np.hstack((label, pooled_output_reshaped.numpy()))

# Save the concatenated data to a CSV file
features_df = pd.DataFrame(data, columns=['label'] + [f'Feature_{i+1}' for i in range(pooled_output_reshaped.shape[1])])
output_file = '../Feature/classification/xl_test1.csv'
features_df.to_csv(output_file, index=False)

# Print save confirmation and the shape of the pooled features
print('已保存')
print(pooled_output_reshaped.shape)
