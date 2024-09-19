import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from utility import Contextual_Attention, ContrastiveLoss
from scipy.stats import pearsonr, spearmanr

# Custom collate function to pad sequences in a batch
def coll_paddding(batch_traindata):
    # Sort by sample length in descending order
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)

    feaQK = []
    feaQK_agv = []
    feaV = []
    train_y = []

    # Append each sample data to the corresponding list
    for data in batch_traindata:
        feaQK.append(data[0])
        feaQK_agv.append(data[1])
        feaV.append(data[2])
        train_y.append(data[3])

    # Get the length of each sample
    data_length = [len(data) for data in feaQK]

    # Create a mask matrix to mark the padded parts
    mask = torch.full((len(batch_traindata), data_length[0]), False).bool()
    for mi, aci in zip(mask, data_length):
        mi[aci:] = True

    # Pad the input features and labels, padding value is 0
    feaQK = torch.nn.utils.rnn.pad_sequence(feaQK, batch_first=True, padding_value=0)
    feaQK_agv = torch.nn.utils.rnn.pad_sequence(feaQK_agv, batch_first=True, padding_value=0)
    feaV = torch.nn.utils.rnn.pad_sequence(feaV, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)

    # Generate a random index sequence to shuffle the data
    random_indices = torch.randperm(feaQK.size(1))

    # Rearrange data based on the random indices
    feaQK_shuffled = feaQK[:, random_indices, :]
    feaQK_agv_shuffled = feaQK_agv[:, random_indices, :]
    feaV_shuffled = feaV[:, random_indices, :]
    train_y_shuffled = train_y[:, random_indices]

    # Remove extra dimensions from labels
    train_y, train_y_shuffled = train_y.squeeze(), train_y_shuffled.squeeze()

    # Calculate comparison labels (ne operation)
    label = torch.ne(train_y, train_y_shuffled).int()

    # Return the processed features, labels, and data lengths
    return feaQK, feaQK_agv, feaV, train_y, feaQK_shuffled, feaQK_agv_shuffled, feaV_shuffled, train_y_shuffled, label, torch.tensor(
        data_length)

# Define a custom dataset class to load data
class BioinformaticsDataset(Dataset):
    def __init__(self, data_path, Feature_QK, Feature_V):
        # Initialize data path and feature files
        self.data_path = data_path
        self.Feature_QK = Feature_QK
        self.Feature_V = Feature_V

    # Read and return data by index
    def __getitem__(self, index):
        # Read QK feature file
        filename_feaQK = self.Feature_QK[index]
        df_feaQK = pd.read_csv(self.data_path + filename_feaQK)
        feaQK = df_feaQK.iloc[:, 1:].values
        if feaQK.dtype == object:
            feaQK = feaQK.astype(float)
        feaQK = torch.tensor(feaQK, dtype=torch.float)

        # Compute the average of QK feature and repeat it to match the original shape
        feaQK_agv = torch.mean(feaQK, dim=0)
        feaQK_agv = feaQK_agv.repeat(feaQK.shape[0], 1)

        # Read V feature file
        filename_feaV = self.Feature_V[index]
        df_feaV = pd.read_csv(self.data_path + filename_feaV)
        feaV = df_feaV.iloc[:, 1:].values
        feaV = torch.tensor(feaV, dtype=torch.float)

        # Extract labels
        label = df_feaQK.iloc[:, 0].values
        label = torch.tensor(label, dtype=torch.float)

        return feaQK, feaQK_agv, feaV, label

    # Return the length of the dataset
    def __len__(self):
        return len(self.Feature_QK)

# Define the CLTAP model
class CLTAP(nn.Module):
    def __init__(self):
        super(CLTAP, self).__init__()
        # Initialize the Contextual Attention layer
        self.ca = Contextual_Attention(q_input_dim=1024, v_input_dim=1024)
        self.relu = nn.ReLU(True)
        # Define convolutional layers
        self.protcnn1 = nn.Conv1d(1024 + 1024 + 1024, 512, 3, padding='same')
        self.protcnn2 = nn.Conv1d(512, 256, 3, padding='same')
        self.protcnn3 = nn.Conv1d(256, 128, 3, padding='same')
        # Define fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        # Define dropout layer
        self.drop = nn.Dropout(0.5)
        # Define sigmoid layer
        self.sigmoid = nn.Sigmoid()

    # Forward pass function
    def forward(self, feaQK, feaQK_agv, feaV, data_length):
        CO_attention = self.ca(feaQK, feaV, data_length)
        return CO_attention

    # Predict module
    def classifier(self, feaQK, feaQK_agv, feaV, data_length):
        with torch.no_grad():
            CO_attention = self.forward(feaQK, feaQK_agv, feaV, data_length)
        # Concatenate QK feature, QK average, and contextual attention
        prot = torch.cat((feaQK, feaQK_agv, CO_attention), dim=2)
        prot = prot.permute(0, 2, 1)
        prot = self.protcnn1(prot)
        prot = self.relu(prot)
        prot = self.protcnn2(prot)
        prot = self.relu(prot)
        prot = self.protcnn3(prot)
        prot = self.relu(prot)
        prot = prot.permute(0, 2, 1)
        x = self.fc1(prot)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.fc4(x)
        # Apply sigmoid to restrict output in (0, 1)
        x = self.sigmoid(x)
        return x

# Prediction function
def predict():
    test_set = BioinformaticsDataset(data_path, featureQK_test, featureV_test)
    # Use DataLoader to load the dataset
    test_loader = DataLoader(dataset=test_set, batch_size=256, num_workers=12, pin_memory=True, persistent_workers=True,
                             collate_fn=coll_paddding)
    model = CLTAP()
    model = model.to(device)
    print("==========================Test RESULT================================")
    # Load pre-trained model weights
    model.load_state_dict(torch.load(model_path + model_name))
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for feaQK1, feaQK1_agv, feaV1, data_y, feaQK2, feaQK2_agv, feaV2, data_y2, label, length in test_loader:
            y_pred = model.classifier(feaQK1.to(device), feaQK1_agv.to(device), feaV1.to(device), length.to(device))
            y_pred = torch.squeeze(y_pred)
            data_y = torch.squeeze(data_y)
            all_outputs.append(y_pred.cpu().numpy())
            all_labels.append(data_y.cpu().numpy())

    # Concatenate all predicted results and true labels
    predict_label = np.concatenate(all_outputs)
    true_label = np.concatenate(all_labels)

    # Calculate and print performance metrics
    pearson_corr, _ = pearsonr(true_label, predict_label)
    spearman_corr, _ = spearmanr(true_label, predict_label)
    mse = metrics.mean_squared_error(true_label, predict_label)
    rmse = mse ** 0.5
    mae = metrics.mean_absolute_error(true_label, predict_label)
    r2 = metrics.r2_score(true_label, predict_label)

    print('Pearson correlation: ', pearson_corr)
    print('Spearman correlation: ', spearman_corr)
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('MAE: ', mae)
    print('R^2 score: ', r2)
    print('<----------------')

    # Save the performance metrics to a CSV file
    metrics_df = pd.DataFrame({
        'Pearson': [pearson_corr],
        'Spearman': [spearman_corr],
        'MSE': [mse],
        'RMSE': [rmse],
        'MAE': [mae],
        'R^2 score': [r2],
    })

    metrics_df.to_csv(model_path + 'CLTAP_reg_metrics.csv', index=False)


# Main function entry
if __name__ == "__main__":
    # Check if GPU is available
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    # Define data and model paths
    data_path = 'Feature/regression/'
    model_path = 'Model/'
    model_name = 'CLTAP_reg.pkl'

    # Feature file names for training and testing
    featureQK_train = ['xl_train.csv']
    featureV_train = ['xxl_train.csv']

    featureQK_test = ['xl_test.csv']
    featureV_test = ['xxl_test.csv']

    # Perform prediction
    predict()
