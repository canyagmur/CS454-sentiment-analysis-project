import os
import re
import yaml
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import TruncatedSVD, PCA

import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"\"", "", text)
        text = text.strip().lower()

        filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        translate_dict = dict((c, " ") for c in filters)
        translate_map = str.maketrans(translate_dict)
        text = text.translate(translate_map)
        return text


class Classifier:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train_and_evaluate(self):
        pass  # To be implemented in subclasses


from sklearn.model_selection import GridSearchCV


class NaiveBayesClassifier(Classifier):
    def __init__(self, X_train, y_train, alphas):
        super().__init__(X_train, y_train)
        self.alphas = alphas
        self.best_alpha = None

    def train_and_evaluate(self, X_val, y_val):
        best_accuracy = 0
        for alpha in self.alphas:
            nb_classifier = MultinomialNB(alpha=alpha)
            nb_classifier.fit(self.X_train, self.y_train)
            y_pred_val = nb_classifier.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred_val)
            print(f'Alpha: {alpha}, Validation Accuracy: {accuracy}')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_alpha = alpha

    def evaluate_on_test(self, X_test, y_test):
        nb_classifier = MultinomialNB(alpha=self.best_alpha)
        nb_classifier.fit(self.X_train, self.y_train)
        y_pred_test = nb_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f'Test Accuracy with Alpha {self.best_alpha}: {accuracy}')
        return accuracy

    def perform_grid_search(self, param_grid, X_val, y_val):
        grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        self.best_alpha = grid_search.best_params_['alpha']
        print("Best alpha:", self.best_alpha)
        y_pred_val = grid_search.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        print(f'Validation Accuracy with Best Alpha: {accuracy}')


class DecisionTreeModel(Classifier):
    def __init__(self, X_train, y_train, max_depth_values, min_samples_split_values, criterion_values):
        super().__init__(X_train, y_train)
        self.max_depth_values = max_depth_values
        self.min_samples_split_values = min_samples_split_values
        self.criterion_values = criterion_values
        self.best_params = None

    def train_and_evaluate(self, X_val, y_val):
        best_accuracy = 0
        for max_depth in self.max_depth_values:
            for min_samples_split in self.min_samples_split_values:
                for criterion in self.criterion_values:
                    dt_classifier = DecisionTreeClassifier(
                        max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion
                    )
                    dt_classifier.fit(self.X_train, self.y_train)
                    y_pred_val = dt_classifier.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred_val)
                    print(
                        f'Max Depth: {max_depth}, Min Samples Split: {min_samples_split}, Criterion: {criterion}, '
                        f'Validation Accuracy: {accuracy}'
                    )
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        self.best_params = {
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'criterion': criterion
                        }

    def evaluate_on_test(self, X_test, y_test):
        dt_classifier = DecisionTreeClassifier(**self.best_params)
        dt_classifier.fit(self.X_train, self.y_train)
        y_pred_test = dt_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f'Test Accuracy with parameters {self.best_params}: {accuracy}')
        return accuracy

    def perform_grid_search(self, param_grid, X_val, y_val):
        grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        self.best_params = grid_search.best_params_
        print("Best parameters:", self.best_params)
        y_pred_val = grid_search.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred_val)
        print(f'Validation Accuracy with Best Parameters: {accuracy}')


class NeuralNetClassifier(Classifier):
    def __init__(self, X_train, y_train, n_epochs, learning_rate, num_layers, hidden_units, activation_function):
        super().__init__(X_train, y_train)
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.model = self.Net(
            self.X_train.shape[1], self.hidden_units, self.num_layers, self.activation_function
        ).float()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    class Net(nn.Module):
        def __init__(self, input_dim, hidden_units, num_layers, activation_function):
            super().__init__()
            self.layers = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(nn.Linear(input_dim, hidden_units))
                else:
                    self.layers.append(nn.Linear(hidden_units, hidden_units))
                if activation_function == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation_function == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                # add more activation functions here if needed
            self.layers.append(nn.Linear(hidden_units, 1))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return torch.sigmoid(x)

    def train_and_evaluate(self, X_val, y_val):
        train_dataloader = self.prepare_dataloader(self.X_train, self.y_train)
        val_dataloader = self.prepare_dataloader(X_val, y_val)

        for epoch in range(self.n_epochs):
            for inputs, targets in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                val_loss = 0
                for inputs, targets in val_dataloader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                print(f'Epoch {epoch+1}, Validation Loss: {val_loss/len(val_dataloader)}')
        return val_loss/len(val_dataloader)

    @staticmethod
    def prepare_dataloader(X, y, batch_size=32):
        tensor_data = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).view(-1, 1).float())
        return DataLoader(tensor_data, batch_size=batch_size, shuffle=True)
    
    def evaluate_on_test(self, X_test, y_test):
        test_dataloader = self.prepare_dataloader(X_test, y_test)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = self.model(inputs)
                predicted = torch.round(outputs)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        test_accuracy = correct / total
        print(f'Test Accuracy: {test_accuracy}')
        return test_accuracy

def load_dataset(filepath):
    return pd.read_pickle(filepath)


def vectorize_data(X_train, X_val, X_test):
    vectorizer = CountVectorizer()  # TfidfVectorizer() #
    X_train_vec = vectorizer.fit_transform(X_train).toarray()  # Convert to dense matrix for NN
    X_val_vec = vectorizer.transform(X_val).toarray()  # Convert to dense matrix for NN
    X_test_vec = vectorizer.transform(X_test).toarray()  # Convert to dense matrix for NN
    return X_train_vec, X_val_vec, X_test_vec


def save_results(output_dir, best_hyperparams, best_accuracy, model_names, accuracies):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.txt'), 'w') as file:
        file.write(f"Best Hyperparameters: Hidden Units={best_hyperparams[0]}, Learning Rate={best_hyperparams[1]}, "
                   f"Test Accuracy={best_accuracy}\n")
        for model_name, accuracy in zip(model_names, accuracies):
            file.write(f"{model_name} Accuracy: {accuracy}\n")
    plt.figure(figsize=(8, 6))
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Performance Comparison')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    #plt.show()


def main():
    # ----------------------- Configuration -----------------------
    with open("config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    if not os.path.exists("output"):
        os.makedirs("output")

    # ----------------------- Data Loading -----------------------
    train = load_dataset('input/train.pkl')
    test = load_dataset('input/test.pkl')
    print("Train shape before preprocessing: ", train.shape)
    print("Test shape before preprocessing: ", test.shape)

    # ----------------------- Data Preprocessing -----------------------
    train['text'] = train['text'].apply(TextPreprocessor.clean_text)
    test['text'] = test['text'].apply(TextPreprocessor.clean_text)

    # ----------------------- Combine train and test sets -----------------------
    combined = pd.concat([train, test])

    # ----------------------- Stratified Sampling -----------------------
    labels = combined['sentiment']
    stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=0.2, random_state=cfg["data"]["random_state"])
    subsampled_dataset_indices, _ = next(stratified_split.split(combined, labels))
    subsampled_data = combined.iloc[subsampled_dataset_indices]

    # Visualizing Class Distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(x='sentiment', data=subsampled_data)
    plt.title('Class Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    plt.savefig('output/class_distribution.png')
    #plt.show()

    # Word Frequency
    words = subsampled_data['text'].str.split()
    words = pd.DataFrame(words.tolist()).stack().value_counts()
    words = words.reset_index().rename(columns={'index': 'word', 0: 'count'})
    print(words.head(30))  # prints the top 10 most frequent words

    # Review Length Distribution
    review_length = subsampled_data['text'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 5))
    sns.histplot(review_length, bins=30)
    plt.title('Review Length Distribution')
    plt.xlabel('Review Length (number of words)')
    plt.ylabel('Frequency')
    plt.savefig('output/review_length_distribution.png')
    #plt.show()

    # ----------------------- Train-Test Split -----------------------
    stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=cfg["data"]["random_state"])
    train_indices, temp_indices = next(stratified_split.split(subsampled_data, subsampled_data['sentiment']))
    temp_data = subsampled_data.iloc[temp_indices]

    stratified_split = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=cfg["data"]["random_state"])
    val_indices, test_indices = next(stratified_split.split(temp_data, temp_data['sentiment']))

    X_train = combined.iloc[train_indices]['text']
    y_train = combined.iloc[train_indices]['sentiment']
    X_val = temp_data.iloc[val_indices]['text']
    y_val = temp_data.iloc[val_indices]['sentiment']
    X_test = temp_data.iloc[test_indices]['text']
    y_test = temp_data.iloc[test_indices]['sentiment']

    # ----------------------- Data Vectorization -----------------------
    X_train_vec, X_val_vec, X_test_vec = vectorize_data(X_train, X_val, X_test)
    print("Train shape after preprocessing: ", X_train_vec.shape)
    print("Validation shape after preprocessing: ", X_val_vec.shape)
    print("Test shape after preprocessing: ", X_test_vec.shape)

    # # ----------------------- Naive Bayes Classifier -----------------------
    nb_classifier = NaiveBayesClassifier(X_train_vec, y_train, cfg["naive_bayes"]["alphas"])
    nb_param_grid = {'alpha': cfg["naive_bayes"]["alphas"]}
    nb_classifier.perform_grid_search(nb_param_grid, X_val_vec, y_val)
    nb_accuracy = nb_classifier.evaluate_on_test(X_test_vec, y_test)

    # ----------------------- Dimensionality Reduction -----------------------
    if cfg["dimensionality_reduction"]["use_pca"]:
        reducer = PCA(n_components=cfg["dimensionality_reduction"]["n_components"])
    else:
        reducer = TruncatedSVD(n_components=cfg["dimensionality_reduction"]["n_components"],
                               random_state=cfg["data"]["random_state"])

    if cfg["dimensionality_reduction"]["use_dim_reduction"]:
        X_train_red = reducer.fit_transform(X_train_vec)
        X_val_red = reducer.transform(X_val_vec)
        X_test_red = reducer.transform(X_test_vec)
    else:
        X_train_red = X_train_vec
        X_val_red = X_val_vec
        X_test_red = X_test_vec

    # ----------------------- Decision Tree Classifier -----------------------
    dt_classifier = DecisionTreeModel(X_train_red, y_train, cfg["decision_tree"]["max_depth_values"],
                                      cfg["decision_tree"]["min_samples_split_values"],
                                      cfg["decision_tree"]["criterion_values"])
    dt_param_grid = {
        'max_depth': cfg["decision_tree"]["max_depth_values"],
        'min_samples_split': cfg["decision_tree"]["min_samples_split_values"],
        'criterion': cfg["decision_tree"]["criterion_values"]
    }
    dt_classifier.perform_grid_search(dt_param_grid, X_val_red, y_val)
    dt_accuracy = dt_classifier.evaluate_on_test(X_test_red, y_test)

    # ----------------------- Neural Net Classifier -----------------------
    param_grid = {
        'hidden_units': cfg["neural_net"]["hidden_units"],
        'learning_rate': cfg["neural_net"]["learning_rate"],
        'num_layers': cfg["neural_net"]["num_layers"],
        'n_epochs': cfg["neural_net"]["n_epochs"],
        'activation_function': cfg["neural_net"]["activation_function"]
    }

    best_loss = np.inf
    best_hyperparams = None

    for hidden_units in param_grid['hidden_units']:
        for learning_rate in param_grid['learning_rate']:
            for num_layers in param_grid['num_layers']:
                for n_epochs in param_grid['n_epochs']:
                    for activation_function in param_grid['activation_function']:
                        neural_net_model = NeuralNetClassifier(X_train_red, y_train.values, n_epochs, learning_rate, num_layers, hidden_units, activation_function)
                        val_loss = neural_net_model.train_and_evaluate(X_val_red, y_val.values)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            best_hyperparams = (hidden_units, learning_rate, num_layers, n_epochs, activation_function)

    test_accuracy = neural_net_model.evaluate_on_test(X_test_red, y_test.values)

    print(f"Best Hyperparameters: Hidden Units={best_hyperparams[0]}, Learning Rate={best_hyperparams[1]}, "
          f"Num Layers={best_hyperparams[2]}, Epochs={best_hyperparams[3]}, Activation Function={best_hyperparams[4]}, "
          f"Test Accuracy={test_accuracy}")
    # ----------------------- Performance Comparison -----------------------
    model_names = ['Naive Bayes', 'Decision Tree', 'Neural Network']
    accuracies = [nb_accuracy, dt_accuracy, test_accuracy]

    save_results('output', best_hyperparams, test_accuracy, model_names, accuracies)


if __name__ == "__main__":
    main()
