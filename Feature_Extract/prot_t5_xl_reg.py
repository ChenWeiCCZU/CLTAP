import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

# Define the sequence length
sequences_length = 9

# Load the model and tokenizer
model_name = 'Rostlab/prot_t5_xl_uniref50'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)

# Read the regression training dataset
df = pd.read_csv('../Dataset/regression_train.csv')

# Get the protein sequences and separate each character (amino acid) with spaces for tokenizer input format
sequences = df['Sequence'].tolist()
sequences = [" ".join(seq) for seq in sequences]

# Get the labels and convert them to a 2D array
label = df['label'].tolist()
label = np.array(label).reshape(-1, 1)

# Set the maximum sequence length for padding and truncation
max_length = sequences_length

# Use the tokenizer to encode sequences, generating tensors that can be input into the model
encoded_inputs = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

# Disable gradient calculation, forward pass to extract the hidden state features from the model
with torch.no_grad():
    outputs = model(**encoded_inputs)

# Get the last hidden state output from the model
sequence_output = outputs.last_hidden_state

# Get batch size, sequence length, and number of features
batch_size, seq_len, features = sequence_output.shape

# Transpose and reshape the model output for pooling operations
sequence_output_reshaped = sequence_output.transpose(1, 2).reshape(batch_size * features, 1, seq_len)

# Perform 1D average pooling with kernel size equal to sequence length
kernel_size = sequences_length
pooling_output = F.avg_pool1d(sequence_output_reshaped, kernel_size=kernel_size, stride=kernel_size)

# Reshape the pooled output and apply mean pooling across the features
pooled_output_reshaped = pooling_output.reshape(batch_size, features, -1).mean(dim=2)

# Concatenate the labels with features into a data matrix
data = np.hstack((label, pooled_output_reshaped.numpy()))

# Save the data to a CSV file
features_df = pd.DataFrame(data, columns=['label'] + [f'Feature_{i+1}' for i in range(pooled_output_reshaped.shape[1])])
output_file = '../Feature/regression/xl_train1.csv'
features_df.to_csv(output_file, index=False)

# Print save confirmation and the shape of the pooled features
print('已保存')
print(pooled_output_reshaped.shape)

##############################
# The process for test data is similar to the training set

# Load the model and tokenizer
model_name = "Rostlab/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5EncoderModel.from_pretrained(model_name)

# Read the test dataset
df = pd.read_csv('../Dataset/regression_test.csv')

# Get protein sequences and preprocess them by adding spaces
sequences = df['Sequence'].tolist()
sequences = [" ".join(seq) for seq in sequences]

# Get the labels and convert them to a 2D array
label = df['label'].tolist()
label = np.array(label).reshape(-1, 1)

# Set the maximum sequence length
max_length = sequences_length

# Encode the test sequences
encoded_inputs = tokenizer(sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

# Forward pass to extract the hidden state features from the model
with torch.no_grad():
    outputs = model(**encoded_inputs)

# Get the last hidden state from the model
sequence_output = outputs.last_hidden_state

# Get batch size, sequence length, and number of features
batch_size, seq_len, features = sequence_output.shape

# Transpose and reshape the output for pooling operations
sequence_output_reshaped = sequence_output.transpose(1, 2).reshape(batch_size * features, 1, seq_len)

# Perform 1D average pooling with kernel size equal to sequence length
kernel_size = sequences_length
pooling_output = F.avg_pool1d(sequence_output_reshaped, kernel_size=kernel_size, stride=kernel_size)

# Reshape the pooled output and apply mean pooling across the features
pooled_output_reshaped = pooling_output.reshape(batch_size, features, -1).mean(dim=2)

# Concatenate the labels with the features into a data matrix
data = np.hstack((label, pooled_output_reshaped.numpy()))

# Save the concatenated data to a CSV file
features_df = pd.DataFrame(data, columns=['label'] + [f'Feature_{i+1}' for i in range(pooled_output_reshaped.shape[1])])
output_file = '../Feature/regression/xl_test1.csv'
features_df.to_csv(output_file, index=False)

# Print save confirmation and the shape of the pooled features
print('已保存')
print(pooled_output_reshaped.shape)
