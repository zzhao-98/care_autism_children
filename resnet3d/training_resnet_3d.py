import pandas as pd
import time
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F

from FOS_dataset import FOS_set, FOS_dataset, transform

if __name__ == '__main__':
    df_train_set = pd.read_csv('train.csv')
    df_val_set = pd.read_csv('val.csv')
    batch_size = 2

    # Define model
    model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
    model.blocks[6].proj = nn.Linear(2304, 4)


    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    # Define dataset
    train_set = FOS_dataset(df_train_set, transform=transform)
    val_set = FOS_dataset(df_val_set, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = model.cuda()
    # criterion = criterion.cuda()

    # Train model
    num_epochs = 10
    start_time = time.time()
    for epoch in range(num_epochs):
        sum_loss = 0
        total = 0
        correct = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            inputs = [i.cuda() for i in inputs]
            labels = labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            sum_loss += loss.item()
            # with torch.no_grad():
            #     print("Model prediction: {}, Ground truth: {}, loss: {}".format(torch.argmax(outputs, axis=1), labels, loss))
            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                      f'Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {sum_loss/(i+1):.4f}, '
                      f'Time: {time.time()-start_time:.3f}s.')
                start_time = time.time()

        # Calculate epoch loss
        mean_loss = sum_loss/len(train_set)
        mean_accuracy = 100 * correct / total
        print('The {} training epoch finished, mean training loss = {:.3f}, training accuracy = {:.3f}%.'.format(epoch+1, mean_loss, mean_accuracy))

        # Save the model after each epoch
        torch.save(model.state_dict(), 'cv_model_balanced.pt')


        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            sum_loss = 0
            for inputs, labels in val_loader:
                inputs = [i.cuda() for i in inputs]
                labels = labels.cuda()
                outputs = model(inputs)

                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                sum_loss += loss

            mean_loss = sum_loss/len(val_set)
            mean_accuracy = 100 * correct / total
            print('The {} validation epoch finished, mean validation loss = {:.3f}, validation accuracy = {:.3f}%.'.format(epoch+1, mean_loss, mean_accuracy))









# if __name__ == '__main__':
#     dir_train_set = r'C:\Users\36394\Study\GWU\PHD in Biomedical Engineer\Research\FOS\Autism_dataset\Vision_dataset\train'
#     dir_val_set = r'C:\Users\36394\Study\GWU\PHD in Biomedical Engineer\Research\FOS\Autism_dataset\Vision_dataset\val'
#     batch_size = 2
#
#     # Define model
#     model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
#     model.blocks[6].proj = nn.Linear(2304, 4)
#
#
#     # Define optimizer and loss function
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
#     criterion = nn.CrossEntropyLoss()
#
#     # Define dataset
#     train_set = FOS_set(dir_train_set, transform=transform)
#     val_set = FOS_set(dir_val_set, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
#
#     model = model.cuda()
#     # criterion = criterion.cuda()
#
#     # Train model
#     num_epochs = 10
#     start_time = time.time()
#     for epoch in range(num_epochs):
#         sum_loss = 0
#         total = 0
#         correct = 0
#         for i, (inputs, labels) in enumerate(train_loader):
#             # Forward pass
#             inputs = [i.cuda() for i in inputs]
#             labels = labels.cuda()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#
#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             with torch.no_grad():
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#
#             sum_loss += loss.item()
#             # with torch.no_grad():
#             #     print("Model prediction: {}, Ground truth: {}, loss: {}".format(torch.argmax(outputs, axis=1), labels, loss))
#             if (i + 1) % 200 == 0:
#                 print(f'Epoch [{epoch + 1}/{num_epochs}], '
#                       f'Step [{i + 1}/{len(train_loader)}], '
#                       f'Loss: {sum_loss/(i+1):.4f}, '
#                       f'Time: {time.time()-start_time:.3f}s.')
#                 start_time = time.time()
#
#         # Calculate epoch loss
#         mean_loss = sum_loss/len(train_set)
#         mean_accuracy = 100 * correct / total
#         print('The {} training epoch finished, mean training loss = {:.3f}, training accuracy = {:.3f}%.'.format(epoch+1, mean_loss, mean_accuracy))
#
#         # Save the model after each epoch
#         torch.save(model.state_dict(), 'cv_model.pt')
#
#
#         # Test the model
#         # In test phase, we don't need to compute gradients (for memory efficiency)
#         with torch.no_grad():
#             correct = 0
#             total = 0
#             sum_loss = 0
#             for inputs, labels in val_loader:
#                 inputs = [i.cuda() for i in inputs]
#                 labels = labels.cuda()
#                 outputs = model(inputs)
#
#                 loss = criterion(outputs, labels)
#
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#                 sum_loss += loss
#
#             mean_loss = sum_loss/len(val_set)
#             mean_accuracy = 100 * correct / total
#             print('The {} validation epoch finished, mean training loss = {:.3f}, training accuracy = {:.3f}%.'.format(epoch+1, mean_loss, mean_accuracy))