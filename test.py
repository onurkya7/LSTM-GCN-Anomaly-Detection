import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class LSTM_GCN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTM_GCN_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.gnn1 = GCNConv(hidden_dim, 128)
        self.gnn2 = GCNConv(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, output_dim)  

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1)) 
        lstm_out = lstm_out[:, -1, :] 
        edge_index = torch.tensor([[0, 1, 2, 3, 4], 
                           [1, 2, 3, 4, 5]], dtype=torch.long)

        gnn_out1 = self.gnn1(lstm_out, edge_index)
        gnn_out1 = self.dropout(gnn_out1)  
        gnn_out2 = self.gnn2(gnn_out1, edge_index)
        gnn_out2 = self.dropout(gnn_out2)  

        output = self.fc(gnn_out2)  
        return output

label_encoder = joblib.load("label_encoder.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

input_dim = tfidf_vectorizer.max_features + 2
hidden_dim = 64
output_dim = len(label_encoder.classes_)
model = LSTM_GCN_Model(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load("lstm_gcn_model.pth"))
model.eval()

# Network traffic data to test
new_csv_file = "wireshark-test.csv"
data = pd.read_csv(new_csv_file, encoding="latin1")

numeric_cols = ['Time', 'Length']
data_numeric = data[numeric_cols].fillna(0).values.astype(np.float32)
data_tfidf = tfidf_vectorizer.transform(data['Info'].fillna("")).toarray()
X_new = np.hstack((data_numeric, data_tfidf)).astype(np.float32)
X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_new_tensor)
    _, predicted = torch.max(outputs, 1)
data['attack_type'] = label_encoder.inverse_transform(predicted.numpy())

# Saving the results to a new file
output_csv_file = "predicted_attack_types.csv"
data.to_csv(output_csv_file, index=False)
print(f"Predictions saved: {output_csv_file}")