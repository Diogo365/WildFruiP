import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import math


def create_dataframe(p, img_path, excel_path, type):
    """!
    Function that creates a dataframe with the features and the labels

    @param p: string with the name of the parameter to be used as label (target)
    @param img_path: path to the images
    @param excel_path: path to the excel file
    @param type: type of the image: prickly_pear or fig

    @return a tuple (data_df, excel) with the dataframe and the excel file
    """
    image_dir = Path(img_path)  # Path to images
    excel = None
    if type == 'prickly_pear':
        excel = pd.read_excel(
            excel_path,
            sheet_name='Prickly Pear', skiprows=11, usecols='B:U')  # Prickly pear parameters
    elif type == 'fig':
        excel = pd.read_excel(
            excel_path,
            sheet_name='Lampa preta', skiprows=6, usecols='B:U')  # Prickly pear parameters

    # drop the first 4 columns (useless)
    excel = excel.drop(excel.columns[[0, 1, 2, 3]], axis=1)
    # hardeness mean
    hardness1 = np.array(excel['Hardeness (N)'])
    hardness2 = np.array(excel['Unnamed: 7'])
    mean_hardness = np.mean([hardness1, hardness2], axis=0)
    excel = excel.drop(excel[['Unnamed: 7']], axis=1)
    excel['Hardeness (N)'] = mean_hardness

    # Color L mean
    color_l1 = np.array(excel['L'])
    color_l2 = np.array(excel['Unnamed: 11'])
    color_l3 = np.array(excel['Unnamed: 12'])
    mean_color_l = np.mean([color_l1, color_l2, color_l3], axis=0)
    excel = excel.drop(excel[['Unnamed: 11', 'Unnamed: 12']], axis=1)
    excel['L'] = np.round(mean_color_l, 2)

    # Color a mean
    color_a1 = np.array(excel['a'])
    color_a2 = np.array(excel['Unnamed: 14'])
    color_a3 = np.array(excel['Unnamed: 15'])
    mean_color_a = np.mean([color_a1, color_a2, color_a3], axis=0)
    excel = excel.drop(excel[['Unnamed: 14', 'Unnamed: 15']], axis=1)
    excel['a'] = np.round(mean_color_a, 2)

    # Color b mean
    color_b1 = np.array(excel['b'])
    color_b2 = np.array(excel['Unnamed: 17'])
    color_b3 = np.array(excel['Unnamed: 18'])
    mean_color_b = np.mean([color_b1, color_b2, color_b3], axis=0)
    excel = excel.drop(excel[['Unnamed: 17', 'Unnamed: 18']], axis=1)
    excel['b'] = np.round(mean_color_b, 2)

    images = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str).sort_values()
    ids = pd.Series(images.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Id').astype(
        int).sort_values()

    parameter = excel[p]
    parameters = pd.Series(ids.apply(lambda x: parameter[x - 1]), name='parameter').astype(float).sort_values()
    data_df = pd.concat([ids, images, parameters], axis=1).reset_index(drop=True)

    return data_df, excel


def normalize_train(p):
    """!
    Normalizes the input array of the training set.

    @:param p (numpy.ndarray): The input array.

    @:return a tuple containing the normalized array, the maximum value, and the minimum value.
    """
    maximum = np.max(p)
    minimum = np.min(p)
    normalized = (p - minimum) / (maximum - minimum)
    return normalized, maximum, minimum


def normalize_test(p, maximum, minimum):
    """!
    Normalizes the input array of the test set.

    @:param p (numpy.ndarray): The input array.
    @:param maximum (float): The maximum value of the training set.
    @:param minimum (float): The minimum value of the training set.

    @:return the normalized array.
    """
    normalized = (p - minimum) / (maximum - minimum)
    return normalized


def denormalize(p, maximum, minimum):
    """!
    Denormalizes the input array.

    @:param p (numpy.ndarray): The input array.
    @:param maximum (float): The maximum value of the training set.
    @:param minimum (float): The minimum value of the training set.

    @:return the denormalized array.
    """
    denormalized = (p * (maximum - minimum)) + minimum
    return denormalized


class AlignedPricklyPearDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """!
        The constructor of the AlignedPricklyPearDataset class.

        @:param dataframe (pandas.DataFrame): The dataframe containing the data.
        @:param transform (torchvision.transforms): The transformation to apply to the images.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.set = set

    def __len__(self):
        """!
        Returns the length of the dataset.

        @:return the length of the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """!
        Returns the item at the given index.

        @:param idx (int): The index of the item to return.

        @:return a tuple containing the image and the corresponding label.
        """
        img_path = self.dataframe.iloc[idx]['Filepath']
        label = self.dataframe.iloc[idx]['parameter']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


def create_mobilenet():
    """!
    Creates a MobileNet model.

    @:return the MobileNet model.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model.classifier[1] = nn.Linear(1280, 1)

    return model


def create_resnet():
    """!
    Creates a ResNet model.

    @:return the ResNet model.
    """
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = nn.Linear(2048, 1)

    return model


# Without KFold Cross Validation
def create_sets(data_df):
    """!
    Creates the training, validation, and test sets.

    @:param data_df (pandas.DataFrame): The dataframe containing the data.

    @:return a tuple containing the training, validation, and test sets, maximum, and minimum of the training set.
    """
    train, test = train_test_split(data_df, train_size=0.8, random_state=1)
    train, val = train_test_split(train, train_size=0.8, random_state=1)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train['parameter'], maximum, minimum = normalize_train(train['parameter'])
    val['parameter'] = normalize_test(val['parameter'], maximum, minimum)
    test['parameter'] = normalize_test(test['parameter'], maximum, minimum)

    train_set = AlignedPricklyPearDataset(train, transform=transform_train)
    val_set = AlignedPricklyPearDataset(val, transform=transform_test)
    test_set = AlignedPricklyPearDataset(test, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    return train_loader, val_loader, test_loader, maximum, minimum


# With KFold Cross Validation
def create_sets2(data_df):
    """!
    Creates the training, validation, and test sets using KFold cross validation.

    @:param data_df (pandas.DataFrame): The dataframe containing the data.

    @:return a tuple containing the training, validation, and test sets, maximum, and minimum of the training set.
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    train_loaders, val_loaders, test_loaders = [], [], []
    maximums, minimums = [], []

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for train_index, test_index in kfold.split(data_df):
        train, test = data_df.iloc[train_index], data_df.iloc[test_index]
        train, val = train_test_split(train, train_size=0.9, random_state=1)

        train['parameter'], maximum, minimum = normalize_train(train['parameter'])
        val['parameter'] = normalize_test(val['parameter'], maximum, minimum)
        test['parameter'] = normalize_test(test['parameter'], maximum, minimum)

        train_set = AlignedPricklyPearDataset(train, transform=transform_train)
        val_set = AlignedPricklyPearDataset(val, transform=transform_test)
        test_set = AlignedPricklyPearDataset(test, transform=transform_test)

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
        maximums.append(maximum)
        minimums.append(minimum)

    return train_loaders, val_loaders, test_loaders, maximums, minimums


def eval_model(p, img_path, excel_path, type, model_path):
    """!
    Evaluates the model, printing the mean of R2, MSE, MAE and MAPE scores of the folds.
    @:param p: parameter to evaluate
    @:param img_path: path to the images
    @:param excel_path: path to the excel file
    @:param type: type of the image: prickly_pear or fig
    @:param model_path: path to the folder containing the models

    @:return None
    """
    data_df, _ = create_dataframe(p, img_path, excel_path, type)
    train_loaders, val_loaders, test_loaders, maxs, mins = create_sets2(data_df)
    R2s = []
    MSEs = []
    MAEs = []
    MAPES = []

    for train_loader, val_loader, test_loader, max, min in zip(train_loaders, val_loaders, test_loaders, maxs, mins):
        model = create_resnet()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(
            torch.load(model_path + '\model' + str(train_loaders.index(train_loader) + 1) + '_' + p + '.pth'))

        y_pred = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                y_pred.extend(outputs.cpu().numpy().flatten())
                y_true.extend(labels.cpu().numpy().flatten())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_pred = denormalize(y_pred, max, min)
        y_true = denormalize(y_true, max, min)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        R2s.append(r2)
        MSEs.append(mse)
        MAEs.append(mae)
        MAPES.append(mape)

    print('Mean R2: ', np.mean(R2s), ' +- ', np.std(R2s))
    print('Mean MSE: ', np.mean(MSEs), ' +- ', np.std(MSEs))
    print('Mean MAE: ', np.mean(MAEs), ' +- ', np.std(MAEs))
    print('Mean MAPE: ', np.mean(MAPES), ' +- ', np.std(MAPES))


def train(parameter, train_iter, img_path, excel_path, type):
    """!
    Trains the model using K-Fold Cross Validation.

    @:param parameter (str): The parameter to be predicted.
    @:param train_iter (int): The number of iterations to train the model.
    @:param img_path (str): The path to the images.
    @:param excel_path (str): The path to the excel file.
    @:param type (str): The type of the image: prickly_pear or fig.

    @:return metrics (dict): A dictionary containing the metrics of the model.
    """
    data_df, _ = create_dataframe(parameter, img_path, excel_path, type)
    train_loaders, val_loaders, test_loaders, maxs, mins = create_sets2(data_df)

    R2s = []
    MSEs = []
    MAEs = []
    MAPES = []
    train_losses = []
    val_losses = []

    for train_loader, val_loader, test_loader, max, min in zip(train_loaders, val_loaders, test_loaders, maxs,
                                                               mins):
        model = create_resnet()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print('Model number: ', train_loaders.index(train_loader) + 1, '/', len(train_loaders), '\n')

        train_loss_values = []
        val_loss_values = []
        min_val_loss = math.inf
        epochs_no_improve = 0
        for epoch in range(100):
            train_epoch_loss = []
            val_epoch_loss = []
            train_loss = 0
            val_loss = 0
            model.train()
            for images, labels in tqdm(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.float(), labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_epoch_loss.append(loss.item())

            train_loss_values.append(sum(train_epoch_loss) / len(train_epoch_loss))

            model.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs.float(), labels.unsqueeze(1).float())
                    val_loss += loss.item()
                    val_epoch_loss.append(loss.item())

                val_loss_values.append(sum(val_epoch_loss) / len(val_epoch_loss))

            # early stopping
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                if epochs_no_improve == 10:
                    print('Early stopping!')
                    if not os.path.exists(type + '_results/' + str(train_iter)):
                        os.makedirs(type + '_results/' + str(train_iter))
                        os.makedirs(type + '_results/' + str(train_iter) + '/max_mins')

                    torch.save(model.state_dict(), type + '_results/' + str(train_iter) + '/model' + str(
                        train_loaders.index(train_loader) + 1) + '_' + parameter.replace('*', '') + '.pth')
                    # save max and min values for denormalization
                    with open(type + '_results/' + str(train_iter) + '/max_mins/' + 'max_min' + str(
                            train_loaders.index(train_loader) + 1) + '_' + parameter.replace('*', '') + '.txt',
                              'w') as f:
                        f.write('Max: ' + str(max) + '\n')
                        f.write('Min: ' + str(min) + '\n')
                    break
            if epoch == 99:
                if not os.path.exists(type + '_results/' + str(train_iter)):
                    os.makedirs(type + '_results/' + str(train_iter))
                    os.makedirs(type + '_results/' + str(train_iter) + '/max_mins')

                torch.save(model.state_dict(), type + '_results/' + str(train_iter) + '/model' + str(
                    train_loaders.index(train_loader) + 1) + '_' + parameter.replace('*', '') + '.pth')
                # save max and min values for denormalization
                with open(type + '_results/' + str(train_iter) + '/max_mins/' + 'max_min' + str(
                        train_loaders.index(train_loader) + 1) + '_' + parameter.replace('*', '') + '.txt',
                          'w') as f:
                    f.write('Max: ' + str(max) + '\n')
                    f.write('Min: ' + str(min) + '\n')

            print(
                f'Epoch: {epoch + 1} \tTraining Loss: {train_loss / len(train_loader):.6f} \tValidation Loss: {val_loss / len(val_loader):.6f} \tLR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Test
        y_pred = []
        y_true = []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                y_pred.extend(outputs.cpu().numpy().flatten())
                y_true.extend(labels.cpu().numpy().flatten())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_pred = denormalize(y_pred, max, min)
        y_true = denormalize(y_true, max, min)

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        R2s.append(r2)
        MSEs.append(mse)
        MAEs.append(mae)
        MAPES.append(mape)
        train_losses.append(train_loss_values)
        val_losses.append(val_loss_values)

        print(f'MAE: {mae}')
        print(f'MSE: {mse}')
        print(f'MAPE: {mape} %')
        print(f'R2: {r2}')

    train_losses_mean = np.array(pd.DataFrame(train_losses).mean(axis=0))
    val_losses_mean = np.array(pd.DataFrame(val_losses).mean(axis=0))

    print('Mean MAE: ', np.mean(MAEs), ' +- ', np.std(MAEs))
    print('Mean MSE: ', np.mean(MSEs), ' +- ', np.std(MSEs))
    print('Mean MAPE: ', np.mean(MAPES), ' +- ', np.std(MAPES))
    print('Mean R2: ', np.mean(R2s), ' +- ', np.std(R2s))

    metrics = {'MAE': np.mean(MAEs), 'MSE': np.mean(MSEs), 'MAPE': np.mean(MAPES), 'R2': np.mean(R2s),
               'Train losses': train_losses_mean, 'Val losses': val_losses_mean}

    return metrics


def hard_samples(p, img_path, excel_path, type, model_path, save_path):
    """!
    Evaluate the model and saves the worst predictions based on the MAE metric.
    @:param p: parameter to evaluate
    @:param img_path: path to the images
    @:param excel_path: path to the excel file
    @:param type: type of the image: prickly_pear or fig
    @:param model_path: path to the model
    @:param save_path: path to save the images

    @:return: None
    """
    df, _ = create_dataframe(p, img_path, excel_path, type)
    _, _, test, maximum, minimum = create_sets(df)

    model = create_resnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in test:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            y_pred.extend(outputs.cpu().numpy().flatten())
            y_true.extend(labels.cpu().numpy().flatten())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = denormalize(y_pred, maximum, minimum)
    y_true = denormalize(y_true, maximum, minimum)

    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    errors = np.abs(y_true - y_pred)  # MAE
    idx = np.argsort(errors)[-4:]  # get the indices of the 4 biggest absolute errors

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in idx:
        print('True: ', y_true[i], 'Pred: ', y_pred[i])
        denormalized_img = (test.dataset[i][0].numpy().transpose(1, 2, 0) * std + mean).clip(0, 1)
        plt.imshow(denormalized_img)
        # save the image
        plt.savefig(save_path + '/' + str(i) + '.png')

    print('R2: ', r2)
    print('MSE: ', mse)
    print('MAE: ', mae)
    print('MAPE: ', mape)


def robust_train(img_path, excel_path, type):
    """!
    Trains the model using K-Fold Cross Validation for all the parameters for ten iterations.

    @:param img_path: path to the images
    @:param excel_path: path to the excel file
    @:param type: type of the image: prickly_pear or fig

    @:return None
    """
    parameters = []
    if type == 'prickly_pear':
        parameters = ['TSS (º Brix)*', 'Hardeness (N)', 'pH', 'mass (g)', 'lenght', 'diameter', 'L', 'a',
                      'b']  # Prickly pear
    elif type == 'fig':
        parameters = ['Hardeness (N)', 'lenght', 'diameter', 'L', 'a', 'b']  # Normal Fig

    for parameter in parameters:
        R2s = []
        MSEs = []
        MAEs = []
        MAPES = []
        train_losses = []
        val_losses = []
        plt.clf()
        for train_iter in range(10):
            print('Train iter: ', train_iter)
            metrics = train(parameter, train_iter, img_path, excel_path, type)
            R2s.append(metrics['R2'])
            MSEs.append(metrics['MSE'])
            MAEs.append(metrics['MAE'])
            MAPES.append(metrics['MAPE'])
            train_losses.append(metrics['Train losses'])
            val_losses.append(metrics['Val losses'])
        train_losses = np.array(pd.DataFrame(train_losses).mean(axis=0))
        val_losses = np.array(pd.DataFrame(val_losses).mean(axis=0))
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(type + '_results/losses_' + parameter.replace('*', '') + '.png')
        with open(type + '_results/losses_' + parameter.replace('*', '') + '.txt', 'w') as f:
            f.write('Train losses: ' + str(train_losses) + '\n')
            f.write('Val losses: ' + str(val_losses) + '\n')

        with open(type + '_results/metrics_' + parameter.replace('*', '') + '.txt', 'w') as f:
            f.write('Mean R2: ' + str(np.mean(R2s)) + ' +- ' + str(np.std(R2s)) + '\n')
            f.write('Mean MSE: ' + str(np.mean(MSEs)) + ' +- ' + str(np.std(MSEs)) + '\n')
            f.write('Mean MAE: ' + str(np.mean(MAEs)) + ' +- ' + str(np.std(MAEs)) + '\n')
            f.write('Mean MAPE: ' + str(np.mean(MAPES)) + ' +- ' + str(np.std(MAPES)) + '\n')


def inference(model_path, img_path, type, max, min):
    """!
    Performs inference on a single image.

    @:param model_path: path to the model
    @:param img_path: path to the image
    @:param type: type of the image: prickly_pear or fig
    @:param max: max value to denormalize
    @:param min: min value to denormalize

    @:return None
    """
    model = create_resnet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    transform_test = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert('RGB')
    image = transform_test(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        output = output.cpu().numpy().flatten()

    if type == 'prickly_pear':
        output = denormalize(output, max, min)
    elif type == 'fig':
        output = denormalize(output, max, min)

    print(output)


def create_parser():
    """!
    Creates the parser for the command line arguments.

    @:return the parser.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--img_path', type=str, required=True,
                              help='Path to the image')
    train_parser.add_argument('--excel_path', type=str, required=True,
                              help='Path to the excel file')
    train_parser.add_argument('--type', type=str, required=True,
                              help='Type of the image: prickly_pear or fig')

    eval_model_parser = subparsers.add_parser('eval-model')
    eval_model_parser.add_argument('-p', '--parameter', type=str, required=True,
                                   help='Parameter to evaluate')
    eval_model_parser.add_argument('--img_path', type=str, required=True,
                                   help='Path to the image')
    eval_model_parser.add_argument('--excel_path', type=str, required=True,
                                   help='Path to the excel file')
    eval_model_parser.add_argument('--type', type=str, required=True,
                                   help='Type of the image: prickly_pear or fig')
    eval_model_parser.add_argument('--model_path', type=str, required=True,
                                   help='Path to the model')

    inference_parser = subparsers.add_parser('inference')
    inference_parser.add_argument('--model', type=str, required=True,
                                  help='Path to the model')
    inference_parser.add_argument('--img', type=str, required=True,
                                  help='Path to the image')
    inference_parser.add_argument('--type', type=str, required=True,
                                  help='Type of the image: prickly_pear or fig')
    inference_parser.add_argument('--max', type=float, required=True,
                                  help='Max value to denormalize')
    inference_parser.add_argument('--min', type=float, required=True,
                                  help='Min value to denormalize')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = create_parser()

    if args.command == 'train':
        robust_train(args.img_path, args.excel_path, args.type)
    elif args.command == 'eval-model':
        eval_model(args.parameter, args.img_path, args.excel_path, args.type, args.model_path)
    elif args.command == 'inference':
        inference(args.model, args.img, args.type, args.max, args.min)
