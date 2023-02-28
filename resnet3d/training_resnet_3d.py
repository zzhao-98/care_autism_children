import pandas as pd
import time
import torch.utils.data
import torch
import torch.nn as nn
import os

from FOS_dataset import FOS_set, FOS_dataset, transform

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"  # using specific gpu to train model.
    df_train_set = pd.read_csv('/home/artmed/Documents/care_autism_children/resnet3d/train.csv')
    df_val_set = pd.read_csv('/home/artmed/Documents/care_autism_children/resnet3d/val.csv')
    batch_size = 24
    device = torch.device('cuda')

    # Define model
    model = torch.hub.load("facebookresearch/pytorchvideo", model="slowfast_r50", pretrained=True)
    model.blocks[6].proj = nn.Linear(2304, 4)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(r'best_cv_model.pt'))

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Define dataset
    train_set = FOS_dataset(df_train_set, transform=transform)
    val_set = FOS_dataset(df_val_set, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Train model
    num_epochs = 100
    max_val_accuracy = 0
    # df_train_log = pd.DataFrame(
    #     columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy', 'time(s)'])
    df_train_log = pd.read_csv('model_training_log.csv')

    for epoch in range(num_epochs):
        train_sum_loss = 0
        total = 0
        correct = 0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            inputs = [i.to(device) for i in inputs]
            labels = labels.to(device)
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
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

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
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
        scheduler.step(val_loss)
        df_train_log.to_csv('model_training_log_1.csv', index=False)

        # Save the model if reach the highest validation accuracy
        if val_mean_accuracy > max_val_accuracy:
            torch.save(model.state_dict(), 'best_cv_model_1.pt')
            print('Highest accuracy, save the model!')
            max_val_accuracy = val_mean_accuracy
