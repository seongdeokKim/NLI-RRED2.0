from copy import deepcopy

import numpy as np
from tqdm import tqdm
import torch
from seqeval.metrics import f1_score, accuracy_score, classification_report


class Trainer():

    def __init__(self, config):
        self.config = config
        self.best_model = None
        self.best_loss = np.inf
        self.best_acc = 0
        self.classification_report = None

    def track_best_loss(self, model, loss):
        if loss <= self.best_loss: # If current epoch returns lower validation loss,
            self.best_loss = loss  # Update lowest validation loss.
            self.best_model = deepcopy(model.state_dict()) # Update best model weights.

    def track_best_acc(self, model, acc):
        if acc >= self.best_acc: # If current epoch returns higher validation accuracy,
            self.best_acc = acc  # Update highest validation accuracy.
            self.best_model = deepcopy(model.state_dict()) # Update best model weights.

    def train(self, model,
              optimizer,
              scheduler,
              train_loader,
              valid_loader,
              index_to_label,
              device):

        for epoch in range(self.config.n_epochs):

            # Put the model into training mode.
            model.train()
            # Reset the total loss for this epoch.
            total_tr_loss = 0

            for step, mini_batch in enumerate(tqdm(train_loader)):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                # You have to reset the gradients of all model parameters
                # before to take another step in gradient descent.
                optimizer.zero_grad()

                # Take feed-forward
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss, logits = outputs['loss'], outputs['logits']

                # Perform a backward pass to calculate the gradients.
                # loss = loss.sum()
                loss = loss.mean()
                loss.backward()
                # track train loss
                total_tr_loss += loss.item()
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()


            # Calculate the average loss over the training data.
            avg_tr_loss = total_tr_loss / len(train_loader)
            # loss={:.4e}
            print('Epoch {} - loss={:.4f}'.format(
                epoch+1,
                avg_tr_loss
            ))
            # Put the model into evaluation mode
            model.eval()
            # Reset the validation loss and accuracy for this epoch.
            total_val_loss = 0
            preds, true_labels = [], []
            for step, mini_batch in enumerate(tqdm(valid_loader)):
                input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
                input_ids, labels = input_ids.to(device), labels.to(device)
                attention_mask = mini_batch['attention_mask']
                attention_mask = attention_mask.to(device)

                # Telling the model not to compute or store gradients,
                # saving memory and speeding up validation
                with torch.no_grad():
                    outputs = model(
                        input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss, logits = outputs['loss'], outputs['logits']

                    loss = loss.mean()
                    total_val_loss += loss.item()

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    labels = labels.to('cpu').numpy()
                    
                    # 2d array
                    for pred in np.argmax(logits, axis=-1):
                        preds.append([index_to_label.get(pred)])
                    for label in labels:
                        true_labels.append([index_to_label.get(label)])

            avg_val_loss = total_val_loss / len(valid_loader)
            avg_val_acc = accuracy_score(true_labels, preds)
            avg_val_f1 = f1_score(true_labels, preds)

            # self.track_best_loss(model, avg_val_loss)
            # print('Validation - loss={:.4f} accuracy={:.4f} f1-score={:.4f} best_loss={:.4f}'.format(
            #     avg_val_loss, avg_val_acc, avg_val_f1, self.best_loss,
            # ))

            self.track_best_acc(model, avg_val_acc)
            print('Validation - loss={:.4f} accuracy={:.4f} f1-score={:.4f} best_acc={:.4f}'.format(
                avg_val_loss, avg_val_acc, avg_val_f1, self.best_acc,
            ))

        model.load_state_dict(self.best_model)

        return model
        
    def test(self, model,
             test_loader,
             index_to_label,
             device):

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss and accuracy for this epoch.
        total_test_loss = 0
        preds, true_labels = [], []
        for step, mini_batch in enumerate(tqdm(test_loader)):
            input_ids, labels = mini_batch['input_ids'], mini_batch['labels']
            input_ids, labels = input_ids.to(device), labels.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss, logits = outputs['loss'], outputs['logits']

                # Calculate the accuracy for this batch of test sentences.
                loss = loss.mean()
                total_test_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                labels = labels.to('cpu').numpy()
                # 2d array
                for pred in np.argmax(logits, axis=-1):
                    preds.append([index_to_label.get(pred)])
                for label in labels:
                    true_labels.append([index_to_label.get(label)])

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = accuracy_score(true_labels, preds)
        avg_test_f1 = f1_score(true_labels, preds)

        print('Test - loss={:.4f} accuracy={:.4f} f1-score={:.4f}'.format(
            avg_test_loss, avg_test_acc, avg_test_f1,
        ))
        print(classification_report(true_labels, preds))

