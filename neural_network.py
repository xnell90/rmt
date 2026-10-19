import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class MNISTClassifier:

    def __init__(self, model, lr=0.01, epochs=20, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.train_losses = []
        self.val_losses = []
        
    def fit(self, X, y, val_perc=10):
        num_rows = int(X.shape[0] * (val_perc / 100))
        
        # Prepare Train Dataset
        self.train_losses = []
        
        X_train = X[:-num_rows]
        y_train = y[:-num_rows]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Prepare Validation Dataset
        self.val_losses = []
        
        X_val = X[-num_rows:]
        y_val = y[-num_rows:]

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        # Define Criterion and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(self.epochs):
            # Training Phase ------------------------------------------------------
            self.model.train()
            
            train_batch_losses = []
            for batch_X, batch_y in train_dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                train_batch_loss = loss.item()
                train_batch_losses.append(train_batch_loss)
                
                loss.backward()
                optimizer.step()

            train_avg_loss = sum(train_batch_losses) / len(train_batch_losses)
            self.train_losses.append(train_avg_loss)

            #Validation Phase ------------------------------------------------------
            self.model.eval()

            val_batch_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                    val_batch_loss = loss.item()
                    val_batch_losses.append(val_batch_loss)

            val_avg_loss = sum(val_batch_losses) / len(val_batch_losses)
            self.val_losses.append(val_avg_loss)
                
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("The model must be fitted before predicting.")
            
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        
        return np.argmax(probabilities, axis=1)