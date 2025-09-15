import torch
from torch import nn
from scipy import sparse
import numpy as np
import pandas as pd
from task3 import sigmoid
import pickle
from task1 import better_tokenize
import time
from task3 import plot_log_likelihood
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
sparse_matrix_root = "/Users/baconjack/Documents/研究生/课程/cse595/matrix.npz"
train_document_rot = "/Users/baconjack/Documents/研究生/课程/cse595/train.csv"
dev_document_root = "/Users/baconjack/Documents/研究生/课程/cse595/dev.csv"

def plot_results(lists, f1scores, weights, step_interval=50):
    steps = [i * step_interval for i in range(1, len(lists[0]) + 1)]
    plt.figure(figsize=(12, 5))
    # Loss Curve
    plt.subplot(1, 2, 1)
    for i, weight in enumerate(weights):
        plt.plot(steps, lists[i], label=f"Learning Rate={weight}")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Dev Loss vs Steps (Different Learning Rate)")
    plt.legend()
    # F1 Curve
    plt.subplot(1, 2, 2)
    for i, weight in enumerate(weights):
        plt.plot(steps, f1scores[i], label=f"Learning Rate={weight}")
    plt.xlabel("Steps")
    plt.ylabel("F1 Score")
    plt.title("Dev F1 vs Steps (Different Learning Rate)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def to_sparse_tensor(sparse_root):
    matrix = sparse.load_npz(sparse_root).tocoo()
    tensor = torch.sparse_coo_tensor(indices=torch.tensor(np.stack([matrix.row, matrix.col])),values = torch.tensor(matrix.data))
    return tensor

class LogisticRegression(nn.Module):
    def __init__(self, len_of_vocab):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(len_of_vocab, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return torch.sparse.mm(x, self.weight) + self.bias

def get_label(y_root):
    documents = pd.read_csv(y_root)
    label = documents["label"]
    label = label.to_numpy().astype(int)
    return label

def evaluate_model(model):
    """
    Vocabulary list address.
    """
    with open("/Users/baconjack/Documents/研究生/课程/cse595/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    loss = nn.BCELoss()
    losses = []
    y_hat = []
    documents = pd.read_csv(dev_document_root)
    texts = documents['generation']
    labels = documents['label']
    model.eval()
    for text , label in zip(texts, labels):
        text_vector = np.zeros(len(vocab))
        tokens = better_tokenize(text)
        for token in tokens:
            if token in vocab:
                id = vocab[token]
                text_vector[id] = 1
        text_vector = torch.tensor(text_vector, dtype=torch.float32)
        text_vector = text_vector.view(1, -1)
        text_vector=text_vector.to_sparse()
        beta_x = model(text_vector)
        prob = torch.sigmoid(beta_x).item()  # 取出 float
        y_hat.append(1 if prob >= 0.5 else 0)
        loss_ = loss(torch.sigmoid(beta_x).squeeze(1), torch.tensor(int(label)).float().unsqueeze(0))
        losses.append(loss_.item())
    #print(f"loss___________{np.sum(losses)}")
    #print(f"f1-score__________{f1_score(labels, y_hat)}")
    model.train()
    return np.sum(losses), f1_score(labels, y_hat)


def train_model(learning_rate, max_epochs, label_root, optimizer_, ):
    text_matrix = to_sparse_tensor(sparse_matrix_root)
    label = get_label(label_root)
    model = LogisticRegression(text_matrix.shape[1])
   # with open(vocab_root, "rb") as f:
        #vocab = pickle.load(f)
    #print("loaded vocab size:", len(vocab))
    model.train()
    loss_function = nn.BCELoss()
    if optimizer_ == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_ == "AdamW":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_ == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    log_likelihood_list_on_eval = []
    for epoch in range(max_epochs):
        indices = np.arange(text_matrix.shape[0])
        np.random.shuffle(indices)
        for count, i in enumerate(indices):
            optimizer.zero_grad()
            y_hat = torch.sigmoid(model(text_matrix[i].unsqueeze(0).float())).squeeze(1)
            loss = loss_function(y_hat, torch.tensor(label[i]).float().unsqueeze(0))
            loss.backward()
            optimizer.step()
            if count%5000 ==0:
                print(count, learning_rate)
                loss_, score = evaluate_model(model)
                if log_likelihood_list_on_eval:
                    if abs((log_likelihood_list_on_eval[-1]-loss_)/loss_) < 1e-4:
                        """
                        After training concludes, directly test the text.csv file and generate the file to be submitted to Kaggle.
                        """
                        #kaggle(model)
                        torch.save(model.state_dict(),"/Users/baconjack/Documents/研究生/课程/cse595/homework/logistic_regression.pth")
                        return log_likelihood_list_on_eval
                    else:
                        log_likelihood_list_on_eval.append(loss_)
                else:
                    log_likelihood_list_on_eval.append(loss_)
        print(f"epoch {epoch}")
    """
    The address of the trained model.
    """
    torch.save(model.state_dict(), "/Users/baconjack/Documents/研究生/课程/cse595/homework/logistic_regression.pth")
    """
    After training concludes, directly test the text.csv file and generate the file to be submitted to Kaggle.
    """
    #kaggle(model)
    return log_likelihood_list_on_eval

def kaggle(model):
    with open("/Users/baconjack/Documents/研究生/课程/cse595/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    y_hat = []
    documents = pd.read_csv("/Users/baconjack/Documents/研究生/课程/cse595/test.student.csv")
    texts = documents['generation']
    ids = documents['id']
    model.eval()
    for text in texts:
        text_vector = np.zeros(len(vocab))
        tokens = better_tokenize(text)
        for token in tokens:
            if token in vocab:
                id = vocab[token]
                text_vector[id] = 1
        text_vector = torch.tensor(text_vector, dtype=torch.float32)
        text_vector = text_vector.view(1, -1)
        text_vector=text_vector.to_sparse()
        beta_x = model(text_vector)
        prob = torch.sigmoid(beta_x).item()  # 取出 float
        y_hat.append(1 if prob >= 0.5 else 0)

    submission = pd.DataFrame({
        "id": ids,
        "label": y_hat
    })
    """
    Storage location for submitted files.
    """
    submission.to_csv("/Users/baconjack/Documents/研究生/课程/cse595/submission_pytorch_with_decay.csv", index=False)

if __name__ == '__main__':
    """
    The following is the training process.
    """
    #tensor = to_sparse_tensor(sparse_matrix_root)
    #lists = []
    #f_1scores = []
    #learning_rates = [0.001]
    #optimizers = ["SGD", "RMSprop", "AdamW"]
    #sparse_matrix_roots = ["/Users/baconjack/Documents/研究生/课程/cse595/matrix_bad.npz", "/Users/baconjack/Documents/研究生/课程/cse595/matrix.npz"]
    #vocab_roots = ["/Users/baconjack/Documents/研究生/课程/cse595/vocab_bad.pkl", "/Users/baconjack/Documents/研究生/课程/cse595/vocab.pkl"]
    #train_model(learning_rate=0.001, max_epochs=30, label_root=train_document_rot, optimizer_="SGD")
    #for learning_rate in learning_rates:
    #    list, f_1score = train_model(learning_rate=learning_rate, max_epochs=20, label_root=train_document_rot, optimizer_="SGD")
    #    lists.append(list)
    #    f_1scores.append(f_1score)
    #plot_results(lists, f_1scores, learning_rates, step_interval=50)
    """
    Below are the top-ranked terms in the beta screening.
    """
    with open("/Users/baconjack/Documents/研究生/课程/cse595/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    #print(vocab)
    model = LogisticRegression(len(vocab))
    """
    The address of the trained model.
    """
    model.load_state_dict(
        torch.load("/Users/baconjack/Documents/研究生/课程/cse595/homework/logistic_regression.pth"))
    model.eval()
    beta = model.weight.detach().cpu().numpy().flatten()
    id2word = {v: k for k, v in vocab.items()}
    top_idx = np.argsort(np.abs(beta))[-20:][::-1]
    top_words = [(id2word[int(i)], beta[i]) for i in top_idx]
    print("Top 10 words with largest |β|:")
    for w, v in top_words:
        print(f"{w}: {v:.4f}")
