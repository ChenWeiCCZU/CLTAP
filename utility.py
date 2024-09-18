import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

# Define the Contextual Attention mechanism class
class Contextual_Attention(nn.Module):
    def __init__(self, q_input_dim, v_input_dim, qk_dim=1024, v_dim=1024):
        super(Contextual_Attention, self).__init__()
        # Define two 1D convolution layers with kernel sizes 3 and 5 for feature extraction
        self.cn3 = nn.Conv1d(q_input_dim, qk_dim, 3, padding='same')
        self.cn5 = nn.Conv1d(q_input_dim, qk_dim, 5, padding='same')
        # Define linear layers for computing Key and Query
        self.k = nn.Linear(qk_dim * 2 + q_input_dim, qk_dim)
        self.q = nn.Linear(q_input_dim, qk_dim)
        # Define linear layer for computing Value
        self.v = nn.Linear(v_input_dim, v_dim)
        # Normalization factor for scaling the attention scores
        self._norm_fact = 1 / torch.sqrt(torch.tensor(qk_dim))

    # Forward function
    def forward(self, feaQK, feaV, seqlengths):
        # Compute Query
        Q = self.q(feaQK)
        # Extract local features using 1D convolution with kernel sizes 3 and 5
        k3 = self.cn3(feaQK.permute(0, 2, 1))
        k5 = self.cn5(feaQK.permute(0, 2, 1))
        # Concatenate original input with the convolutional features
        feaQK = torch.cat((feaQK, k3.permute(0, 2, 1), k5.permute(0, 2, 1)), dim=2)
        # Compute Key
        K = self.k(feaQK)
        # Compute Value
        V = self.v(feaV)
        # Compute attention scores and apply softmax normalization
        atten = masked_softmax((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact, seqlengths)
        # Compute output based on attention scores
        output = torch.bmm(atten, V)
        # Return the sum of the computed output and the Value
        return output + V

# Define Contrastive Loss class for measuring the similarity between samples
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.8):
        super(ContrastiveLoss, self).__init__()
        # Set margin value to control the minimum distance between dissimilar samples
        self.margin = margin

    # Forward function
    def forward(self, output1, output2, label):
        # Remove extra dimensions
        output1, output2, label = output1.squeeze(), output2.squeeze(), label.squeeze()

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(output1, output2)
        # euclidean_distance = F.pairwise_distance(output1, output2)
        # manhattan_distance = torch.abs(output1 - output2).sum(dim=1)
        # chebyshev_distance, _ = torch.max(torch.abs(output1 - output2), dim=1)

        # Compute contrastive loss, which is divided into two parts:
        #    This term minimizes the squared cosine similarity, pushing similar samples closer.
        #    This term maximizes the distance between dissimilar samples, ensuring the margin.
        loss_contrastive = torch.mean((1 - label) * torch.pow(cosine_similarity, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2))

        # Return the computed loss value
        return loss_contrastive

def save_prob_label(probs,labels,filename):
    #data={'probs':probs,'labels':labels}
    probs = np.array(probs)
    labels = np.array(labels)
    data = np.hstack((probs.reshape(-1, 1), labels.reshape(-1, 1)))
    names = ['probs', 'labels']
    Pd_data = pd.DataFrame(columns=names, data=data)
    Pd_data.to_csv(filename, header=True, index=False)

def create_src_lengths_mask(batch_size: int, src_lengths):

    max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    # returns [batch_size, max_seq_len]
    return (src_indices < src_lengths).int().detach()

def masked_softmax(scores, src_lengths, src_length_masking=True):
    #scores [batchsize,L*L]
    if src_length_masking:
        bsz, src_len, max_src_len = scores.size()
        # compute masks
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        src_mask = torch.unsqueeze(src_mask, 2)
        #print('scr_mask',src_mask)
        #scores=scores.permute(0,2,1)
        # Fill pad positions with -inf
        scores=scores.permute(0,2,1)
        scores = scores.masked_fill(src_mask == 0, -np.inf)
        scores = scores.permute(0, 2, 1)
        #print('scores',scores)
    return F.softmax(scores.float(), dim=-1)



