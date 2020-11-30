from time import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from baseline_model import Baseline
from mask_dataset import MaskDataSet
import argparse


def load_data(batch_size, td, tl, vd, vl):
    tds = MaskDataSet(td, tl)
    vds = MaskDataSet(vd, vl)
    train_loader = DataLoader(tds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(vds, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader


def load_model(lr, model_type, loss_fnc_type="mse"):
    model = Baseline()
    if model_type == "resnet":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        ct = 0
        for name, child in model.named_children():
            if ct < 4:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
            ct += 1


    loss_fnc = nn.MSELoss()
    if loss_fnc_type != "mse":
        loss_fnc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, loss_fnc, optimizer


def accuracy(predict, label, loss_fnc_type):
    # refactor to use matrix math for speed
    total_corr = 0
    if loss_fnc_type != "mse":
        predict = torch.argmax(predict, dim=1).detach().numpy()
        label = torch.argmax(label, dim=1).detach().numpy()
        total_corr = np.sum(predict == label)
    else:
        print(predict)
        print(label)
        for i in range(len(predict)):
            this_pred = predict[i].detach().numpy()
            this_label = label[i].detach().numpy()
            pred_max = np.argmax(this_pred)

            if this_label[pred_max] == 1:
                total_corr = total_corr + 1

    return total_corr / len(predict)


def evaluate(model, val_loader, loss_fnc, loss_fnc_type):
    total_corr = 0
    total = 0
    nbatch = 0
    total_loss = 0

    for img, label in val_loader:
        if nbatch + 1 == len(val_loader):
            break
        predict = model(img)
        loss = get_loss(predict, label, loss_fnc_type, loss_fnc)
        total_loss = total_loss + loss.item()
        total_corr = total_corr + accuracy(predict, label, loss_fnc_type) * len(predict)
        total = total + len(predict)
        nbatch = nbatch + 1

    return float(total_corr / total), float(total_loss / nbatch)


def test_eval(model, loss_fnc, testd, testl):
    total_corr = 0
    total = 0
    total_loss = 0
    print(testd)
    for i in range(len(testd)):
        total = total + 1
        predict = model(testd[i].unsqueeze(0)).squeeze(0)
        label = testl[i]
        if label[np.argmax(predict.detach().numpy())] == 1:
            total_corr = total_corr + 1
        total_loss = total_loss + loss_fnc(predict, label).item()

    return total_corr / total, total_loss / total


def get_loss(input, target, loss_fnc_type, loss_fnc):
    if loss_fnc_type != "mse":
        target = torch.argmax(target, dim=1)
    return loss_fnc(input=input, target=target)

def main(args):
    # Hyperparameters
    nepochs = args.epochs
    lr = args.lr
    bs = args.batch_size
    seed = args.seed
    model_type = args.model_type
    loss_fnc_type = args.loss_fnc_type

    # 70, 10, 20 train, validation, test split
    # This is probably not the best way to do this but it was the quickest to get working
    images = torch.load('images.pt')
    oh_labels = torch.load('oh_labels.pt')

    td, tvd, tl, tvl = train_test_split(images, oh_labels, test_size=0.3, random_state=0)
    vd, testd, vl, testl = train_test_split(tvd, tvl, test_size=float(2 / 3), random_state=0)

    # Training loop
    torch.manual_seed(seed)

    # Initialize model, loss_fnc, optimizer
    model, loss_fnc, optimizer = load_model(lr, model_type, loss_fnc_type)
    e = 0

    # Initialize data loader
    train_iter, val_iter = load_data(bs, td, tl, vd, vl)

    # For plotting
    tlossRec = []
    templossRec = []
    tempaccRec = []
    vlossRec = []
    taccRec = []
    vaccRec = []
    eRec = []

    # Time
    start_time = time()
    for i in range(nepochs):
        model.train()
        for img, label in train_iter:
            optimizer.zero_grad()
            predict = model(img)
            loss = get_loss(predict, label, loss_fnc_type, loss_fnc)
            loss.backward()
            optimizer.step()
            # Track training loss and accuracy throughout
            templossRec.append(loss.item())
            tempaccRec.append(accuracy(predict, label, loss_fnc_type))
        print(i)

        tlossRec.append(sum(templossRec) / len(templossRec))
        taccRec.append(sum(tempaccRec) / len(tempaccRec))
        model.eval()
        vacc, vloss = evaluate(model, val_iter, loss_fnc, loss_fnc_type)
        vaccRec.append(vacc)
        vlossRec.append(vloss)
        eRec.append(e)
        e = e + 1
        templossRec = []
        tempaccRec = []

    stop_time = time()
    if nepochs > 0:
        print("Total time:", stop_time - start_time)
        print("Final training accuracy: ", taccRec[-1])
        print("Final validation accuracy: ", vaccRec[-1])
        print("Final training loss: ", tlossRec[-1])
        print("Final validation loss: ", vlossRec[-1])

        plt.plot(eRec, taccRec, color='red')
        plt.plot(eRec, vaccRec, color='blue')
        plt.title('Accuracy (nEp:{}, L_r:{}, batch:{}, seed:{})'.format(nepochs, lr, bs, seed))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Training', 'Validation'])

        plt.plot(eRec, tlossRec, color='red')
        plt.plot(eRec, vlossRec, color='blue')
        plt.title('Loss (nEp:{}, L_r:{}, batch:{}, seed:{})'.format(nepochs, lr, bs, seed))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'])

    testacc, testloss = test_eval(model, loss_fnc, testd, testl)

    print("Test accuracy:", testacc, "\n")
    print("Test loss:", testloss)

    torch.save(model, 'baseline_maskless.pt2')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model-type', type=str, default='baseline',
                        help="Model type: baseline, resnet (Default: baseline)")
    parser.add_argument('--loss-fnc-type', type=str, default="mse")
    parser.add_argument('--overfit', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args)
