import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, Fully
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'fully':
        model = Fully()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # Output model summary
    # summary(model, (1, 28, 28))

    Acc_train, Loss_train = [], []
    Acc_val, Loss_val = [], []

    # Run any number of epochs you want
    n_epoch = 10
    Epoch = [i for i in range(1, 1+n_epoch)]
    for epoch in range(n_epoch):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch         
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))

        # Store acc && loss to plot
        Acc_train.append(acc)
        Loss_train.append(ave_loss)

        ################
        ## Validation ##
        ################
        model.eval()
        # TODO
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0

        with torch.no_grad():
            for batch, (x, label) in enumerate(val_loader, 1):
                # Put input tensor to GPU if it's available
                if use_cuda:
                    x, label = x.cuda(), label.cuda()
                # Forward input tensor through your model
                out = model(x)
                # Calculate loss
                loss = criterion(out, label)

                # Calculate the training loss and accuracy of each iteration
                total_loss += loss.item()
                _, pred_label = torch.max(out, 1)
                total_cnt += x.size(0)
                correct_cnt += (pred_label == label).sum().item()

                # Show the training information
                if batch % 500 == 0 or batch == len(val_loader):
                    acc = correct_cnt / total_cnt
                    ave_loss = total_loss / batch         
                    print ('Validation batch index: {}, valid loss: {:.6f}, acc: {:.3f}'.format(
                        batch, ave_loss, acc))

        # Store acc && loss to plot
        Acc_val.append(acc)
        Loss_val.append(ave_loss)

        model.train()

    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Plot Learning Curve
    # TODO
    fig, (acc_curve, loss_curve) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.5)

    ## Accuracy curve
    acc_curve.set_title('Accuracy curve')
    acc_curve.plot(Epoch, Acc_train, color='c', label='Training Accuracy')
    acc_curve.plot(Epoch, Acc_val, color='r', label='Validation Accuracy')
    acc_curve.legend(loc='lower right')

    ## Loss curve
    loss_curve.set_title('Loss curve')
    loss_curve.plot(Epoch, Loss_train, color='c', label='Training Loss')
    loss_curve.plot(Epoch, Loss_val, color='r', label='Validation Loss')
    loss_curve.legend(loc='upper right')

    ## Output the curves
    fig.savefig('curves_' + model.name() + '.png')
    fig.show()