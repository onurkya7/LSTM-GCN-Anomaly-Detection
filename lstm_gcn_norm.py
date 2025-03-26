import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file_paths = {
    "dataset/benign/benign-1.csv": "benign",
    "dataset/benign/benign-2.csv": "benign",
    "dataset/benign/benign-3.csv": "benign",
    "dataset/benign/benign-4.csv": "benign",
    "dataset/benign/benign-5.csv": "benign",
    "dataset/brute_force/brute_force.csv": "brute_force",
    "dataset/brute_force/mysql_brute_force.csv": "brute_force",
    "dataset/brute_force/redis_brute_force.csv": "brute_force",
    "dataset/brute_force/SMB_brute_force-2.csv": "brute_force",
    "dataset/brute_force/SMB_brute_force.csv": "brute_force",
    "dataset/probe/probe.csv": "probe",
    "dataset/exploit/cve-2020.csv": "exploit",
    "dataset/flood/http_slowloris.csv": "flood",
    "dataset/flood/udp_flood.csv": "flood",
    "dataset/flood/http-flood.csv": "flood",
    "dataset/flood/syn-flood.csv": "flood",
    "dataset/flood/ntp-amplification.csv": "flood",
    "dataset/flood/icmp_flood.csv": "flood",
    "dataset/malware/ransomware/cerber.csv": "malware",
    "dataset/malware/ransomware/globelmposter.csv": "malware",
    "dataset/malware/spyware/agent_tesla.csv": "malware",
    "dataset/malware/spyware/form_book.csv": "malware",
    "dataset/malware/spyware/redline.csv": "malware",
    "dataset/malware/trojan/chthonic.csv": "malware",
    "dataset/malware/trojan/lokibot.csv": "malware",
    "dataset/malware/trojan/ratinfected.csv": "malware",
    "dataset/malware/trojan/squirrel_waffle.csv": "malware",
    "dataset/malware/trojan/delf-banker.csv": "malware",
}

train_dataframes = []
test_dataframes = []

for file_path, attack_type in file_paths.items():
    if attack_type == "exploit": 
        df = pd.read_csv(file_path, nrows=5000, encoding='latin1')
        df['attack_type'] = "exploit"
        train_dataframes.append(df.iloc[:4000])  
        test_dataframes.append(df.iloc[-1000:])  
    elif attack_type == "brute_force": # 1000 * 5 = 5000 (Brute Force)
        df = pd.read_csv(file_path, nrows=1000, encoding='latin1')
        df['attack_type'] = "brute_force"
        train_dataframes.append(df.iloc[:800])  
        test_dataframes.append(df.iloc[-200:])  
    elif attack_type == "probe": # 5000 (Probe)
        df = pd.read_csv(file_path, nrows=5000, encoding='latin1')
        df['attack_type'] = "probe"
        train_dataframes.append(df.iloc[:4000])  
        test_dataframes.append(df.iloc[-1000:])  
    elif attack_type == "benign": # 1000 * 5 = 5000 (Benign)
        df = pd.read_csv(file_path, nrows=1000, encoding='latin1')
        df['attack_type'] = "benign"
        train_dataframes.append(df.iloc[:800])  
        test_dataframes.append(df.iloc[-200:]) 
    elif attack_type == "malware": # 500 * 10 = 5000 (Malware)
        df = pd.read_csv(file_path, nrows=500, encoding='latin1')
        df['attack_type'] = "malware"
        train_dataframes.append(df.iloc[:400])  
        test_dataframes.append(df.iloc[-100:])  
    elif attack_type == "flood": 
        df = pd.read_csv(file_path, nrows=800, encoding='latin1')
        df['attack_type'] = "flood"
        train_dataframes.append(df.iloc[:672])  
        test_dataframes.append(df.iloc[-167:])  

train_data = pd.concat(train_dataframes, ignore_index=True)
test_data = pd.concat(test_dataframes, ignore_index=True)

def print_class_distribution(data, dataset_name="Dataset"):
    class_counts = data['attack_type'].value_counts()
    print(f"{dataset_name} - Class Distribution:")
    for attack_type, count in class_counts.items():
        print(f"  {attack_type}: {count} sample")
    print("\n")

print_class_distribution(train_data, "Training Data")
print_class_distribution(test_data, "Test Data")

label_encoder = LabelEncoder()
train_data['attack_type'] = label_encoder.fit_transform(train_data['attack_type'])
test_data['attack_type'] = label_encoder.transform(test_data['attack_type'])

numeric_cols = ['Time', 'Length']
tfidf_vectorizer = TfidfVectorizer(max_features=500)

train_info_features = tfidf_vectorizer.fit_transform(train_data['Info'].fillna("")).toarray()
test_info_features = tfidf_vectorizer.transform(test_data['Info'].fillna("")).toarray()

scaler = StandardScaler()

X_train_numeric = scaler.fit_transform(train_data[numeric_cols].fillna(0).values)
X_test_numeric = scaler.transform(test_data[numeric_cols].fillna(0).values)

X_train = np.hstack((X_train_numeric, train_info_features))
X_test = np.hstack((X_test_numeric, test_info_features))

y_train = train_data['attack_type'].values
y_test = test_data['attack_type'].values

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

class AttackDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return features, label

train_dataset = AttackDataset(X_train, y_train)
test_dataset = AttackDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# LSTM-GCN Model 
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

input_dim = X_train.shape[1]
hidden_dim = 16
output_dim = len(label_encoder.classes_)
model = LSTM_GCN_Model(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

def train_epoch(model, train_loader):
    model.train()
    total_loss = 0
    correct = 0
    for features, labels in train_loader:
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / len(train_loader.dataset)
    return total_loss / len(train_loader), accuracy

def evaluate_epoch(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for features, labels in test_loader:
            output = model(features)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy

# Training loop
num_epochs = 120
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(1, num_epochs + 1):
    train_loss, train_accuracy = train_epoch(model, train_loader)
    test_loss, test_accuracy = evaluate_epoch(model, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print("-" * 40)

# ==========================================================
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        output = model(features)
        _, predicted = torch.max(output, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(y_true, y_pred)
class_names = label_encoder.inverse_transform(sorted(set(y_true)))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', xticks_rotation='vertical', ax=ax)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Confusion Matrix", fontsize=13)
plt.subplots_adjust(top=0.8)  
plt.tight_layout()
plt.show()

def plot_roc_curve(model, test_loader, num_classes, class_names):
    model.eval()
    
    y_true = []
    y_score = [[] for _ in range(num_classes)]  
    
    with torch.no_grad():
        for features, labels in test_loader:
            output = model(features)
            y_true.extend(labels.cpu().numpy())
            for i in range(num_classes):
                y_score[i].extend(output[:, i].cpu().numpy())
    
    y_true_bin = label_binarize(y_true, classes=[i for i in range(num_classes)])
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[i])
        roc_auc = auc(fpr, tpr)  
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')  
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

class_names = label_encoder.inverse_transform(sorted(set(y_true)))  
num_classes = len(label_encoder.classes_)
plot_roc_curve(model, test_loader, num_classes, class_names)

plt.figure(figsize=(12, 5))
steps = range(5, num_epochs + 1, 5)  

plt.subplot(1, 2, 1)
plt.plot(steps, [train_losses[i - 1] for i in steps], label="Training Loss")
plt.plot(steps, [test_losses[i - 1] for i in steps], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(steps, [train_accuracies[i - 1] for i in steps], label="Training Accuracy")
plt.plot(steps, [test_accuracies[i - 1] for i in steps], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    specificity_list = []
    conf_matrix = confusion_matrix(y_true, y_pred)
    for i in range(len(conf_matrix)):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_list.append(tn / (tn + fp))
    specificity = np.mean(specificity_list)

    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1-score: {f1:.4f}")

calculate_metrics(y_true, y_pred)

# ==========================================================
import joblib

joblib.dump(label_encoder, 'label_encoder.pkl')
torch.save(model.state_dict(), "lstm_gcn_model.pth")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
