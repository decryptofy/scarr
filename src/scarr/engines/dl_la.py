from .engine import Engine
from ..modeling.dl_models import DL_Models as dlm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
from math import floor

np.seterr(divide='ignore', invalid='ignore')


class DL_LA(Engine):

    def __init__(self, model_type, train_float, num_epochs) -> None:
        # remember construction parameters
        self.model_type = model_type
        self.train_float = train_float
        self.num_epochs = num_epochs
        # initialize values needed
        self.samples_len = 0
        self.batch_size = 0
        self.traces_len = 0
        self.batches_num = 0
        self.counted_batches = 0
        self.data_dtype = None
        self.sensitivity = None
        # validation values
        self.accuracy = 0
        self.actual_labels = None
        self.pred_labels = None
        self.predicted_classes = None


    def populate(self, container):
        # initialize dimensional variables
        self.samples_len = container.min_samples_length
        self.traces_len = container.min_traces_length
        self.batch_size = container.data.batch_size
        self.batches_num = int(self.traces_len/self.batch_size)
        # assign per-tile train and validation data
        for tile in container.tiles:
            (tile_x, tile_y) = tile
        # config batches
        container.configure(tile_x, tile_y, [0])
        container.configure2(tile_x, tile_y, [0])


    def fetch_training_batch(self, container, i):
        batch1 = container.get_batch_index(i)[-1]
        batch2 = container.get_batch_index2(i)[-1]
        current_data = np.concatenate((batch1, batch2), axis=0)
        label1 = np.zeros(len(batch1))
        label2 = np.ones(len(batch2))
        current_labels = np.concatenate((label1, label2), axis=0)
        current_labels = np.eye(2)[current_labels.astype(int)]  # one-hot encode labels
        return current_data, current_labels


    def fetch_validation_batch(self, container, i, batch_size):
        batch1 = container.get_batch_index(i)[-1]
        batch2 = container.get_batch_index2(i)[-1]
        current_data = np.concatenate((batch1, batch2), axis=0)
        label1 = np.zeros(batch_size)
        label2 = np.ones(batch_size)
        current_labels = np.concatenate((label1, label2), axis=0)
        return current_data, current_labels


    def train_model(self, container):
        # start model
        if self.model_type == 'MLP':
            self.model = dlm.MLP(self.samples_len)
        elif self.model_type == 'CNN':
            self.model = dlm.CNN(self.samples_len)
        else:
            print("Invalid model type entered")
            return
        # define loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # begin traininig
        for epoch in range(self.num_epochs):
            self.counted_batches = 0
            epoch_loss = 0.0
            accumulated_gradients = None
            workers = 2
            if not container.fetch_async:
                workers = 1

            with ThreadPoolExecutor(max_workers=workers) as executor:
                future = executor.submit(self.fetch_training_batch, container, 0)
                for i in range(floor((self.traces_len/self.batch_size)*self.train_float)):
                    # wait for prev batch 
                    current_data, current_labels = future.result()
                    # begin async next batch fatch
                    if container.fetch_async:
                        future = executor.submit(self.fetch_training_batch, container, i+1)
                
                    # allocate batch tensors
                    batch_X = torch.tensor(current_data, dtype=torch.float32, requires_grad=True)
                    batch_Y = torch.tensor(current_labels, dtype=torch.float32)
                    # DL 
                    # foward pass
                    outputs = self.model(batch_X)
                    # calculate loss
                    loss = self.criterion(outputs, batch_Y)
                    # backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()

                    # accumulate gradients for sensitivity analysis
                    if accumulated_gradients is None:
                        accumulated_gradients = batch_X.grad.data.clone()
                    else:
                        accumulated_gradients += batch_X.grad.data
                    
                    self.optimizer.step()
                    
                    # accumulate loss
                    epoch_loss += loss.item()
                    self.counted_batches += 1

                    # begin sync next batch fatch
                    if not container.fetch_async:
                        future = executor.submit(self.fetch_training_batch, container, i+1)

                # sensitivity stuff
                average_gradients = accumulated_gradients / self.counted_batches if self.counted_batches > 0 else 0
                self.sensitivity = average_gradients.abs().mean(dim=0)
                # final training stuff
                average_loss = epoch_loss / self.counted_batches if self.counted_batches > 0 else 0
                print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {average_loss}")


    def validate_model(self, container):
        # get new data
        X_new = np.empty((2*(self.batches_num-self.counted_batches)*self.batch_size, self.samples_len), dtype=self.data_dtype)
        Y_test = np.empty((2*(self.batches_num-self.counted_batches)*self.batch_size), dtype=self.data_dtype)
        workers = 2
        if not container.fetch_async:
            workers = 1
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future = executor.submit(self.fetch_validation_batch, container, self.counted_batches, self.batch_size)

            for i in range(self.counted_batches, self.batches_num):
                # wait for prev batch 
                current_data, current_labels = future.result()
                # begin async next batch fatch
                if container.fetch_async:
                    future = executor.submit(self.fetch_validation_batch, container, i+1, self.batch_size)

                start_idx = (i-self.counted_batches) * self.batch_size
                end_idx = start_idx + 2 * self.batch_size
                X_new[start_idx:end_idx] = current_data
                Y_test[start_idx:end_idx] = current_labels

                # begin sync next batch fatch
                if not container.fetch_async:
                    future = executor.submit(self.fetch_validation_batch, container, i+1, self.batch_size)

        # save labels
        self.actual_labels = Y_test[:]
        # make new data into tensors
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

        # set model to evaluation mode
        self.model.eval()

        # make predictions
        with torch.no_grad():
            predictions = self.model(X_new_tensor)

        # convert raw outputs to probabilites
        probabilities = torch.softmax(predictions, dim=1)

        # get predicted class
        self.predicted_classes = torch.argmax(probabilities, dim=1)
        # copy locally for sensitivity
        self.pred_labels = self.predicted_classes.numpy(force=True)

        # print predictions
        # print("Predicted classes:\n", self.predicted_classes)
        # calculate accuracy
        correct_predictions = (self.predicted_classes == torch.tensor(Y_test)).sum().item()
        self.accuracy = correct_predictions / len(Y_test)
        # print("Accuracy:", accuracy)
        
        # clean print message
        print("Made", self.predicted_classes.size(dim=0), "predictions with", f"{self.accuracy:.2%}", "% accuracy using the", self.model_type, "model.")


    def run(self, container, model_building=False, model_validation=False):
        if model_building:
            # initialize vars and arrays
            self.populate(container)
            # being pytorch stuff
            device = (
                # "cpu"
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            #  start model
            if self.model_type == 'MLP':
                self.model = dlm.MLP(self.samples_len).to(device)
            elif self.model_type == 'CNN':
                self.model = dlm.CNN(self.samples_len).to(device)
            else:
                print(">> Invalid model type entered")
                return
            # begin training
            self.train_model(container)
        # begin validating if true
        if model_validation:
            self.validate_model(container)


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

     
    def load_model(self, container, path):
        # populate initial values as if for training
        self.populate(container)
        self.counted_batches = floor((self.traces_len/self.batch_size)*self.train_float)
        # select and load model
        if self.model_type == 'MLP':
            self.model = dlm.MLP(self.samples_len)
        elif self.model_type == 'CNN':
            self.model = dlm.CNN(self.samples_len)
        else:
            print("Invalid model type entered")
            return
        self.model.load_state_dict(torch.load(path))
        print("Loaded", self.model_type, "model from memory")


    def print_info(self):
        print("> trace dimentions:", self.traces_len, self.samples_len)


    def get_accuracy(self):
        return self.accuracy

    def get_sensitivity(self):
        return self.sensitivity.numpy()
    