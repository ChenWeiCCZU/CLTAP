# CLTAP
A TAP-Binding Peptide Prediction Method Based on Contrastive Learning and Co-Attention Mechanism

The Transporter associated with Antigen Processing (TAP) transports peptides to the endoplasmic reticulum, a necessary step for determining CD8 T cell epitopes. CLTAP is a novel prediction method designed to improve the accuracy of TAP-binding peptide prediction by integrating contrastive learning and attention mechanisms. This study introduces contrastive learning for TAP-binding peptide prediction for the first time, significantly enhancing prediction performance by optimizing the model’s ability to distinguish between different samples. Additionally, the attention mechanism introduced in the study effectively identifies and focuses on key features within peptide sequences, further enhancing the model’s capability to handle complex biological sequence data. Through the combination of these two technologies, CLTAP has demonstrated superior predictive performance compared to existing methods across multiple datasets. Evaluations on test datasets show that CLTAP surpasses current SOTA methods by 6.32% in ACC, 15.26% in MCC, 4.48% in F1-Score, and 4.13% in AUC. It provides an efficient and reliable tool for TAP-binding peptide research and application.

# 1. Requirements
Python >= 3.9.18

torch = 2.1.1

pandas = 2.2.2

scikit-learn = 1.3.2

scipy = 1.13.0

ProtT5-XL-UniRef50 model and ProtT5-XXL-UniRef50 model

# 2. Description
The proposed CLTAP method is implemented using Python on Torch for predicting TAP binding peptides. The CLTAP model uses contrastive learning combined with contextual co attention mechanism and adopts a staged training strategy to improve the accuracy of TAP binding peptide prediction.
# 3 Datasets
classification_DS868.csv: this file contains 604 TAP binding peptides and 264 non TAP binding peptides for classification tasks

regression_DS613.csv: This file contains 384 TAP binding peptides and 229 non TAP binding peptides for regression tasks

# 4. How to Use
## 4.1 Set up environment for ProtTrans
Set ProtTrans follow procedure from https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main and https://huggingface.co/Rostlab/prot_t5_xxl_uniref50.

## 4.2 Extract features
Extract Prot-T5-xl feature: cd to the CLTAP/ Feature_Extract dictionary, and run "python3 prot_t5_xl.py" and "python3 prot_t5_xl_reg.py", the Prot-T5-xl feature will be extracted.

Extract Prot-T5-xxl feature: cd to the CLTAP/ Feature_Extract dictionary, and run "python3 prot_t5_xxl.py" and "python3 prot_t5_xxl_reg.py", the Prot-T5-xxl feature will be extracted.

## 4.3 Prediction
Classification Task: cd to the CLTAP dictionary,and run "python3 CLTAP_cla.py" to predict TAP binding peptides.

Regression Task: cd to the CLTAP dictionary,and run "python3 CLTAP_reg.py" to predict the binding affinity of the peptide to TAP.