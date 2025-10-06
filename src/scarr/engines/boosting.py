# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# This Source Code Form is "Incompatible With Secondary Licenses", as
# defined by the Mozilla Public License, v. 2.0.

from .engine import Engine
from ..modeling.dl_models import DL_Models as dlm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from math import floor
from tqdm import tqdm


class DLLABoostingEnsemble:
    def __init__(self, base_model_fn, n_estimators=3, lr=0.001, device=None):
        self.base_model_fn = base_model_fn
        self.n_estimators = n_estimators
        self.lr = lr
        self.sensitivity = None
        self.models = []
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.BCELoss()
        self.epoch_batches = []
        self.current_stage = 0
        self.epochs_per_stage = 1
        self.ready = False
        self.curr_model = None
        self.optimizer = None
        self._fixed_epoch_batches = None


    def begin_stage(self):
        self.curr_model = self.base_model_fn().to(self.device)
        self.optimizer = torch.optim.Adam(self.curr_model.parameters(), lr=1e-3)
        self.ready = True


    def update(self, X_batch, y_batch, log_batch=True):
        if not self.ready:
            self.begin_stage()

        X_batch = X_batch.to(self.device).float()
        y_batch = y_batch.to(self.device).long()

        # compute current ensemble prediction (logits)
        with torch.no_grad():
            ensemble_logits = torch.zeros(X_batch.size(0), 2).to(self.device)
            for model in self.models:
                ensemble_logits += self.lr * model(X_batch)

            if len(self.models) > 0:
                ensemble_logits /= (self.lr * len(self.models))
            else:
                ensemble_logits = torch.full_like(y_batch, 0.5)

        # compute residual pseudo-targets (added stability)
        target = (y_batch - ensemble_logits).detach()
        target = (target + 1.0) / 2.0
        target = target.clamp(0.0, 1.0)

        # train current model
        self.curr_model.train()
        pred_logits = self.curr_model(X_batch)

        eps = 1e-5
        pred_logits = pred_logits.clamp(eps, 1. - eps)

        loss = self.criterion(pred_logits, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def end_stage(self):
        # save only model's state_dict
        self.models.append(copy.deepcopy(self.curr_model).eval())
        torch.cuda.empty_cache()  # helps prevent CUDA OOM
        self.current_stage += 1
        self.ready = False


    def finish_training(self):
        if self._fixed_epoch_batches is None:
            # copy batches to CPU and detach
            self._fixed_epoch_batches = [
                (X.detach().cpu(), y.detach().cpu())
                for (X, y) in self.epoch_batches
            ]

        for _ in range(self.n_estimators - self.current_stage):
            self.begin_stage()
            for _ in range(self.epochs_per_stage):
                for X, y in self._fixed_epoch_batches:
                    self.update(X, y, log_batch=False)
            self.end_stage()


    def predict(self, X):
        X = X.to(self.device).float()
        logits = torch.zeros((X.size(0), 2), device=self.device)
        for model in self.models:
            logits += self.lr * model(X)
        probs = F.softmax(logits, dim=1)
        
        # return class index
        return torch.argmax(probs, dim=1) 
    

    def compute_sensitivity(self, X_input):
        self.curr_model.eval()
        X_input = X_input.to(self.device).float()
        X_input.requires_grad = True

        # accumulate logits from all ensemble models
        ensemble_logits = torch.zeros((X_input.size(0), 2), device=self.device)
        for model in self.models:
            model.eval()
            logits = model(X_input)
            ensemble_logits += self.lr * logits
        ensemble_logits /= (self.lr * len(self.models))

        # use the logit difference for class sensitivity
        class_diff = ensemble_logits[:, 1] - ensemble_logits[:, 0]

        # backpropagate to input features
        class_diff.sum().backward()
        gradients = X_input.grad

        # aggregate absolute gradients across all samples
        sensitivity_scores = gradients.abs().mean(dim=0).detach().cpu().numpy()
        return sensitivity_scores
    



class Boosting(Engine):
    def __init__(self, model_type, num_estimators, train_float, num_epochs) -> None:
        # construction parameters
        self.model_type = model_type
        self.train_float = train_float
        self.num_epochs = num_epochs
        self.num_estimators = num_estimators
        # initialize values needed
        self.samples_len = 0
        self.batch_size = 0
        self.traces_len = 0
        self.batches_num = 0
        self.counted_batches = 0
        self.data_dtype = None
        self.sensitivity = None
        self.sens_tensor = None
        self.p_value = 0
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
    

    def train_ensemble(self, container):
        num_batches = floor((self.traces_len / self.batch_size) * self.train_float)
        print(f"Training {self.num_estimators} estimators on {num_batches} batches")

        # feed batches
        for i in tqdm(range(num_batches), desc="Processing batches"):
            data_np, labels_oh = self.fetch_training_batch(container, i)
            X = torch.tensor(data_np, dtype=torch.float32)
            y = torch.tensor(labels_oh, dtype=torch.float32)
            self.model.update(X, y)
            self.counted_batches += 1

        self.model.finish_training()

        print(f"Finished training {self.num_estimators} models.")


    def validate_ensemble(self, container):
        num_val_batches = self.batches_num - self.counted_batches
        print(f'Validating on {num_val_batches} batches')

        X_new = np.empty((2*num_val_batches*self.batch_size, self.samples_len))
        Y_test = np.empty((2*num_val_batches*self.batch_size))
      
        for i in tqdm(range(num_val_batches), desc="Processing batches"):
                current_data, current_labels = self.fetch_validation_batch(container, i + int(self.batches_num * self.train_float), self.batch_size)

                start_idx = i * self.batch_size
                end_idx = start_idx + 2 * self.batch_size
                X_new[start_idx:end_idx] = current_data
                Y_test[start_idx:end_idx] = current_labels
        
        # save labels
        self.actual_labels = Y_test[:]
        # make new data into tensors
        X_new_tensor = torch.tensor(X_new, dtype=torch.float32)

        # make predictions
        preds = self.model.predict(X_new_tensor)
        preds = preds.cpu().numpy(force=True)
        
        # calculate accuracy
        correct_predictions = np.sum(preds == Y_test)
        self.accuracy = correct_predictions / len(Y_test)
        
        print(f"Made {preds.shape[0]} predictions with {self.accuracy:.2%} accuracy using the {self.model_type} model.")

        # sensitivity stuff
        self.sens_tensor = X_new_tensor


    def run(self, container, model_building=False, model_validation=False):
        # training
        if model_building:
            self.populate(container)
            # initialize boosting ensemble
            self.model = DLLABoostingEnsemble(
                base_model_fn=lambda: dlm.eMLP(self.samples_len),
                n_estimators=self.num_estimators,
                lr=0.001,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.train_ensemble(container)
        # validation
        if model_validation:
            self.validate_ensemble(container)


    def get_sensitivity(self):
        self.sensitivity = self.model.compute_sensitivity(self.sens_tensor)       
        return self.sensitivity
    

    def get_accuracy(self):
        return self.accuracy


    # ===== p-value and leakage stuff =====
    def binom_log_pmf(self,k, n, p):
        if p == 0.0: return float('-inf') if k > 0 else 0.0
        if p == 1.0: return float('-inf') if k < n else 0.0
        return (
            math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p)
            + (n - k) * math.log(1 - p)
        )


    def logsumexp(self, log_probs):
        max_log = max(log_probs)
        return max_log + math.log(sum(math.exp(lp - max_log) for lp in log_probs))


    def get_log10_binom_tail(self, k_min, n, p):
        if k_min > n: return float('-inf')  # log(0)

        log_probs = [self.binom_log_pmf(k, n, p) for k in range(k_min, n + 1)]
        log_p_value = self.logsumexp(log_probs)
        log10_p_value = log_p_value / math.log(10)  # convert ln(p) to log10(p)
        return -log10_p_value


    def get_leakage(self, p_th=1e-5):
        M = self.traces_len - self.counted_batches * self.batch_size
        sM = int(np.floor(self.accuracy * M))
        sM = max(0, min(sM, M))

        # compute -log10(p)
        neg_log10_p = self.get_log10_binom_tail(sM, M, 0.5)
        self.p_value = 10 ** (-neg_log10_p)  # only for comparison/display

        if self.p_value <= p_th:
            print(f"Leakage detected: p-value ≈ {self.p_value:.2e}, -log10(p) ≈ {neg_log10_p:.2f}")
        else:
            print(f"No significant leakage: p-value ≈ {self.p_value:.2e}, -log10(p) ≈ {neg_log10_p:.2f}")

        return self.p_value, neg_log10_p