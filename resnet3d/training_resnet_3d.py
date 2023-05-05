import pandas as pd
import time
import torch.utils.data
import torch
import torch.nn as nn
import os

from FOS_dataset import FOS_set, rand_transform, transform

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"  # using specific gpu to train model.
    df_train_set = pd.read_csv(r"C:\Users\36394\PycharmProjects\care_autism_children\resnet3d\train.csv")
    df_val_set = pd.read_csv(r"C:\Users\36394\PycharmProjects\care_autism_children\resnet3d\val.csv")
    batch_size = 12
    device = torch.device('cuda')

    # Define model
    model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
    # model.blocks[6].proj = nn.Linear(2304, 6)
    model.blocks[6].proj = nn.Sequential(
        nn.Linear(in_features=2304, out_features=400),
        nn.ReLU(),
        nn.Linear(in_features=400, out_features=3)
    )
    model = model.to(device)
    # model = nn.DataParallel(model).to(device)
    # model.load_state_dict(torch.load(r'best_cv_model.pt'))

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.MultiLabelSoftMarginLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    # Define dataset
    vision_classes = ['C+', 'PN', 'EA']
    train_set = FOS_set(df_train_set, list_caring_labels=vision_classes, transform=rand_transform)
    val_set = FOS_set(df_val_set, list_caring_labels=vision_classes, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Train model
    num_epochs = 100
    max_val_accuracy = 0
    df_train_log = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'time(s)'])
    # df_train_log = pd.read_csv('model_training_log.csv')

    for epoch in range(num_epochs):
        train_sum_loss = 0
        total = 0
        correct = 0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            inputs = [i.to(device) for i in inputs]
            labels = labels.to(device)
            # labels = labels.to(device)
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                predicted = outputs > 0.5
                labels = labels == 1
                total += labels.numel()
                correct += (predicted == labels).sum().item()

            train_sum_loss += train_loss.item()

        # Calculate epoch loss
        train_mean_loss = train_sum_loss / len(train_set)
        train_mean_accuracy = 100 * correct / total

        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            val_sum_loss = 0
            for inputs, labels in val_loader:
                inputs = [i.to(device) for i in inputs]
                labels = labels.to(device)
                outputs = model(inputs)

                val_loss = criterion(outputs, labels)

                predicted = outputs > 0.5
                labels = labels == 1
                total += labels.numel()
                correct += (predicted == labels).sum().item()
                val_sum_loss += val_loss.item()

            val_mean_loss = val_sum_loss / len(val_set)
            val_mean_accuracy = 100 * correct / total
            time_cost = time.time()-start_time
            print('{}/{} epoch completed. Train loss:{:.3f}; validation loss:{:.3f}; train accuracy: {:.3f}%; '
                  'validation accuracy: {:.3f}%; time cost: {:.3f}s.'
                  .format(epoch+1, num_epochs,
                          train_mean_loss, val_mean_loss, train_mean_accuracy, val_mean_accuracy,
                          time_cost))
            df_train_log.loc[len(df_train_log.index)] = [epoch + 1, train_mean_loss, train_mean_accuracy, val_mean_loss,
                                                         val_mean_accuracy, time_cost]
        scheduler.step()
        df_train_log.to_csv('model_training_log_balanced.csv', index=False)

        # Save the model if reach the highest validation accuracy
        if val_mean_accuracy > max_val_accuracy:
            torch.save(model.state_dict(), 'best_cv_model_balanced.pt')
            print('Highest accuracy, save the model!')
            max_val_accuracy = val_mean_accuracy
