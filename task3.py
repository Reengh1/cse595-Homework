import numpy as np
from task1 import better_tokenize
import pickle
from scipy import sparse
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def sigmoid(x, beta):
    return 1 / (1 + np.exp(-np.dot(x, beta)))

def log_likelihood(beta, X, y):
    return np.sum(y*(X.dot(beta))-np.log(1+np.exp(X.dot(beta))))

def logistic_regression(learning_rate, X, Y, num_steps):
    log_likelihood_list = []

    beta = np.zeros(X.shape[1]+1)
    bias = np.ones((X.shape[0],1))
    X = hstack([X, csr_matrix(bias)])
    #print(beta)
    #print(X)
    for k in range(num_steps):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        for count, i in enumerate(indices):
            #if count % 10000 == 0:
                #print(log_likelihood(beta, X, Y))
                #log_likelihood_list.append(log_likelihood(beta, X, Y))
            row = X.getrow(i).toarray().ravel()
            y_hat = sigmoid(row, beta)
            partial_ll = (y_hat - Y[i]) * row
            beta = beta - learning_rate * partial_ll
        print("finished step", k, "--log_liklihood:", log_likelihood(beta, X, Y))
        log_likelihood_list.append(log_likelihood(beta, X, Y))
        if k>0:
            if abs((log_likelihood_list[k-1]-log_likelihood_list[k])/log_likelihood_list[k]) < 1e-5:
                return beta, log_likelihood_list

    return beta, log_likelihood_list

def predict(beta, texts):
    result = []
    with open('/Users/baconjack/Documents/研究生/课程/cse595/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    for text in texts:
        text_vector = np.zeros(len(vocab)+1)
        tokens = better_tokenize(text)
        for token in tokens:
            if token in vocab:
                id = vocab[token]
                text_vector[id] = 1
        text_vector[-1] = 1
        result.append(sigmoid(text_vector, beta))
    return np.array(result)

def plot_log_likelihood(log_likelihood_list, step_interval=1):
    steps = [i * step_interval for i in range(len(log_likelihood_list))]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, log_likelihood_list, linestyle='-', linewidth=2, color='b', label="Loss")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Progress of Logistic Regression")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_model(csv_root, beta_root):
    documents = pd.read_csv(csv_root)
    documents = documents[['generation', 'label']]
    texts = documents['generation']
    label = documents['label'].to_numpy().astype(int)
    beta = np.load(beta_root)
    results = predict(beta, texts)
    y_hat = (results > 0.5).astype(int)
    f1 = f1_score(label, y_hat)
    print("f1_score:", f1)

def kaggle(csv_root, beta_root):
    documents = pd.read_csv(csv_root)
    texts = documents['generation']
    ids = documents['id']
    beta = np.load(beta_root)
    results = predict(beta, texts)
    y_hat = (results > 0.5).astype(bool)
    submission = pd.DataFrame({
        "id": ids,
        "label": y_hat
    })
    submission.to_csv("/Users/baconjack/Documents/研究生/课程/cse595/submission___test__.csv", index=False)
    print(submission)



if __name__ == '__main__':
    """
    The following commented-out sections pertain to the training process.
    """
    #matrix = sparse.load_npz('/Users/baconjack/Documents/研究生/课程/cse595/matrix.npz')
    #matrix = matrix.toarray()
    #csv_root = "/Users/baconjack/Documents/研究生/课程/cse595/train.csv"
    #documents = pd.read_csv(csv_root)
    #labels = documents["label"]
    #labels = labels.astype(int).to_numpy()
    #beta, list = logistic_regression(5e-4, matrix, labels, 500)
    #np.save("/Users/baconjack/Documents/研究生/课程/cse595/beta.npy", beta)
    #plot_log_likelihood(list)
    """
    Below is the code for generating Kaggle competition submission files.
    """
    dev_csv_root = "/Users/baconjack/Documents/研究生/课程/cse595/dev.csv"
    text_csv_root = "/Users/baconjack/Documents/研究生/课程/cse595/test.student.csv"
    beta_root  = "/Users/baconjack/Documents/研究生/课程/cse595/beta.npy"
    kaggle(text_csv_root, beta_root)


