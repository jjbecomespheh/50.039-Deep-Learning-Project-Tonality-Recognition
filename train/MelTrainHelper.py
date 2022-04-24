import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class TrainHelper():
    
    def loss_func(self, predictions, targets):
        loss = nn.CrossEntropyLoss()
        return loss(input=predictions,target=targets)

    def train_step(self, X, Y, model, optimizer):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X) #
        predictions = torch.argmax(output_softmax,dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        # compute loss
        loss = self.loss_func(output_softmax, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy*100


    def validate(self,X,Y,model):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X) #, attention_weights_norm
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            loss = self.loss_func(output_softmax,Y)
        return loss.item(), accuracy*100, predictions
    
    def validate_top3(self,X,Y, model):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax,dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            loss = self.loss_func(output_logits,Y)
        return loss.item(), accuracy*100, predictions, output_softmax

    def model_train(self, model, epochs, batch_size, X_train, Y_train, X_val, Y_val, device, lr, optimizer="Adam", patience = 1000000):
        EPOCHS=epochs
        DATASET_SIZE = X_train.shape[0]
        BATCH_SIZE = batch_size
        print('Number of trainable params: ',sum(p.numel() for p in model.parameters()) )
        if optimizer == "SGD":
            OPTIMIZER = torch.optim.SGD(model.parameters(),lr=lr, weight_decay=1e-3, momentum=0.8)
        elif optimizer == "Adam":
            OPTIMIZER = torch.optim.Adam(model.parameters(),lr=lr)
           
        losses=[]
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        count = 0
        min_val_loss = 100000
        for epoch in range(EPOCHS):
            # schuffle data
            ind = np.random.permutation(DATASET_SIZE)
            X_train = X_train[ind,:,:,:]
            Y_train = [Y_train[i] for i in ind] 
            epoch_acc = 0
            epoch_loss = 0
            iters = int(DATASET_SIZE / BATCH_SIZE)
            for i in range(iters):
                batch_start = i * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, DATASET_SIZE)
                actual_batch_size = batch_end-batch_start
                X = X_train[batch_start:batch_end,:,:,:]
                Y = Y_train[batch_start:batch_end]
                X_tensor = torch.tensor(X,device=device).float()
                Y_tensor = torch.tensor(Y, dtype=torch.long,device=device)
                loss, acc = self.train_step(X_tensor,Y_tensor, model, OPTIMIZER)
                epoch_acc += acc*actual_batch_size/DATASET_SIZE
                epoch_loss += loss*actual_batch_size/DATASET_SIZE
                print(f"\r Epoch {epoch}: iteration {i}/{iters}",end='')
            X_val_tensor = torch.tensor(X_val,device=device).float()
            Y_val_tensor = torch.tensor(Y_val,dtype=torch.long,device=device)
            val_loss, val_acc, _ = self.validate(X_val_tensor,Y_val_tensor, model)
            losses.append(epoch_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            train_accuracies.append(epoch_acc)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
            elif count > patience:
                print('')
                print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")
                break
            else:
                count +=1

            print('')
            print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{val_loss:.4f}, val_acc:{val_acc:.2f}%")
        return model, losses, train_accuracies, val_losses, val_accuracies
