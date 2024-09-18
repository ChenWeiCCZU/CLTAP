from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from utility import save_prob_label, Contextual_Attention, ContrastiveLoss

# Custom collate function for padding the data
def coll_paddding(batch_traindata):
    # Sort the batch data by length (from longest to shortest)
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)

    feaQK = []
    feaQK_agv = []
    feaV = []
    train_y = []

    # Append data from the batch to respective lists
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

    # Pad each input feature (padding value is 0)
    feaQK = torch.nn.utils.rnn.pad_sequence(feaQK, batch_first=True, padding_value=0)
    feaQK_agv = torch.nn.utils.rnn.pad_sequence(feaQK_agv, batch_first=True, padding_value=0)
    feaV = torch.nn.utils.rnn.pad_sequence(feaV, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)

    # Generate a random index sequence and shuffle the data
    random_indices = torch.randperm(feaQK.size(1))

    # Reorder each variable based on random indices
    feaQK_shuffled = feaQK[:, random_indices, :]
    feaQK_agv_shuffled = feaQK_agv[:, random_indices, :]
    feaV_shuffled = feaV[:, random_indices, :]
    train_y_shuffled = train_y[:, random_indices]

    # 计算对比标签 (XOR 操作)
    # Compute the label transformation (XOR operation)
    label = (train_y ^ train_y_shuffled)

    # Return the processed features, labels, and data length
    return feaQK, feaQK_agv, feaV, train_y, feaQK_shuffled, feaQK_agv_shuffled, feaV_shuffled, train_y_shuffled, label, torch.tensor(data_length)

# Define a custom dataset class
class BioinformaticsDataset(Dataset):
    def __init__(self, data_path, Feature_QK, Feature_V):
        # Initialize data path and feature files
        self.data_path = data_path
        self.Feature_QK = Feature_QK
        self.Feature_V = Feature_V

    # Get data by index
    def __getitem__(self, index):
        # Read the QK feature file
        filename_feaQK = self.Feature_QK[index]
        df_feaQK = pd.read_csv(self.data_path + filename_feaQK)
        feaQK = df_feaQK.iloc[:, 1:].values
        if feaQK.dtype == object:
            feaQK = feaQK.astype(float)
        feaQK = torch.tensor(feaQK, dtype=torch.float)

        # Compute the average of QK feature and repeat it to match the original shape
        feaQK_agv = torch.mean(feaQK, dim=0)
        feaQK_agv = feaQK_agv.repeat(feaQK.shape[0], 1)

        # Read the V feature file
        filename_feaV = self.Feature_V[index]
        df_feaV = pd.read_csv(self.data_path + filename_feaV)
        feaV = df_feaV.iloc[:, 1:].values
        feaV = torch.tensor(feaV, dtype=torch.float)

        # Extract labels
        label = df_feaQK.iloc[:, 0].values
        label = torch.tensor(label, dtype=torch.long)

        return feaQK, feaQK_agv, feaV, label

    # Length of the dataset
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
        self.fc4 = nn.Linear(16, 2)
        # Define dropout layer
        self.drop = nn.Dropout(0.5)

    # Forward pass function
    def forward(self, feaQK, feaQK_agv, feaV, data_length):
        CO_attention = self.ca(feaQK, feaV, data_length)
        return CO_attention

    # Predict module
    def classifier(self, feaQK, feaQK_agv, feaV, data_length):
        with torch.no_grad():
            CO_attention = self.forward(feaQK, feaQK_agv, feaV, data_length)
        # Concatenate features and perform convolution and classification
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
        return x

# Prediction function
def predict():
    test_set = BioinformaticsDataset(data_path, featureQK_test, featureV_test)
    # Use DataLoader to load the dataset
    test_loader = DataLoader(dataset=test_set, batch_size=256, num_workers=12, pin_memory=True, persistent_workers=True, collate_fn=coll_paddding)
    model = CLTAP()
    model = model.to(device)
    print("==========================Test RESULT================================")
    # Load pre-trained model weights
    model.load_state_dict(torch.load(model_path + model_name))
    model.eval()
    arr_probs = []
    arr_labels = []
    arr_labels_hyps = []
    with torch.no_grad():
        # Perform prediction on test data
        for feaQK1, feaQK1_agv, feaV1, data_y, feaQK2, feaQK2_agv, feaV2, data_y2, label, length in test_loader:
            y_pred = model.classifier(feaQK1.to(device), feaQK1_agv.to(device), feaV1.to(device), length.to(device))
            y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, length.to('cpu'), batch_first=True)
            y_pred = y_pred.data
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            arr_probs.extend(y_pred[:, 1].to('cpu'))
            y_pred = torch.argmax(y_pred, dim=1).to('cpu')
            data_y = torch.nn.utils.rnn.pack_padded_sequence(data_y, length, batch_first=True)
            arr_labels.extend(data_y.data)
            arr_labels_hyps.extend(y_pred)

    # Calculate various evaluation metrics
    auc = metrics.roc_auc_score(arr_labels, arr_probs)
    accuracy = metrics.accuracy_score(arr_labels, arr_labels_hyps)
    balanced_accuracy = metrics.balanced_accuracy_score(arr_labels, arr_labels_hyps)
    mcc = metrics.matthews_corrcoef(arr_labels, arr_labels_hyps)
    tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = 2 * tp / (2 * tp + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    youden = sensitivity + specificity - 1

    # Output the evaluation results
    print('acc ', accuracy)
    print('balanced_accuracy ', balanced_accuracy)
    print('tn, fp, fn, tp ', tn, fp, fn, tp)
    print('MCC ', mcc)
    print('sensitivity ', sensitivity)
    print('specificity ', specificity)
    print('precision ', precision)
    print('recall ', recall)
    print('f1score ', f1score)
    print('youden ', youden)
    print('auc', auc)
    print('<----------------')

    # Save the prediction results and evaluation metrics
    name = 'CLTAP'
    save_prob_label(arr_probs, arr_labels, model_path + 'CLTAP_probs.csv')
    print('<----------------save to csv finish')
    metrics_df = pd.DataFrame({
        'Metric': ['Name', 'AUC', 'Accuracy', 'Balanced Accuracy', 'TN', 'FP',
                   'FN', 'TP', 'MCC', 'Sensitivity', 'Specificity',
                   'F1 Score', 'Recall', 'Precision', 'Youden'],
        'Value': [name, auc, accuracy, balanced_accuracy, tn, fp, fn, tp, mcc, sensitivity, specificity,
                  f1score, recall, precision, youden]
    })
    metrics_df = metrics_df.T
    metrics_df.to_csv(model_path + 'CLTAP_metrics.csv', header=False, index=False)

# Main function entry
if __name__ == "__main__":
    # Check if GPU is available
    cuda = torch.cuda.is_available()
    torch.cuda.set_device(0)
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")

    # Define data and model paths
    data_path = 'Feature/classification/'
    model_path = 'Model/'
    model_name = 'CLTAP_cla.pkl'

    # Feature file names for training and testing
    featureQK_train = ['xl_train.csv']
    featureV_train = ['xxl_train.csv']

    featureQK_test = ['xl_test.csv']
    featureV_test = ['xxl_test.csv']

    # Perform prediction
    predict()
