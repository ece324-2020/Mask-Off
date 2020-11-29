from time import time
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from mask_dataset import MaskDataSet

imagef = torchvision.datasets.ImageFolder('/content/drive/MyDrive/Mask Off/datasets_cv',
                                          transform=transforms.ToTensor())
data_loader = DataLoader(imagef, batch_size=30)


for img, label in data_loader:
    print(img.shape)
    break

# Normalize Data
all_im = []
for i in range(len(imagef)):
    all_im = all_im + [imagef[i][0]]

all_im = torch.stack(all_im)

chan0 = all_im[:, 0, :, :]
chan1 = all_im[:, 1, :, :]
chan2 = all_im[:, 2, :, :]
chan0_mean = chan0.mean()
print("Chan0 mean is:", chan0_mean)
chan1_mean = chan1.mean()
print("Chan1 mean is:", chan1_mean)
chan2_mean = chan2.mean()
print("Chan2 mean is:", chan2_mean)
chan0_std = chan0.std()
print("Chan0 std is:", chan0_std)
chan1_std = chan1.std()
print("Chan1 std is:", chan1_std)
chan2_std = chan2.std()
print("Chan2 std is:", chan2_std)

means = [chan0_mean, chan1_mean, chan2_mean]
stds = [chan0_std, chan1_std, chan2_std]


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(means, stds)])

fullImages = torchvision.datasets.ImageFolder('/content/drive/MyDrive/Mask Off/datasets_cv', transform=transform)

switcher = {
    0: [1.0, 0.0, 0.0],
    1: [0.0, 1.0, 0.0],
    2: [0.0, 0.0, 1.0],
}

start_time = time()

images = []
oh_labels = []
for i in range(len(fullImages)):
    images = images + [fullImages[i][0]]
    oh_labels = oh_labels + [torch.tensor(switcher[fullImages[i][1]])]
    if i % 100 == 0:
        print(i)

images = torch.stack(images)
oh_labels = torch.stack(oh_labels)
masked_ds = MaskDataSet(images, oh_labels)

end_time = time()
total_time = end_time - start_time
print("total time: (s)", total_time)


# Save model (if desired)
torch.save(images, 'images.pt')
torch.save(oh_labels, 'oh_labels.pt')
# torch.save(masked_ds, "masked_v2.pt")

# Check if normalized
data_loader = DataLoader(masked_ds, batch_size=6000)

for img, label in data_loader:
    # Seperate the colour channels
    chan0 = img[:, 0, :, :]
    chan1 = img[:, 1, :, :]
    chan2 = img[:, 2, :, :]
    # Compute mean
    chan0_mean = chan0.mean()
    chan1_mean = chan1.mean()
    chan2_mean = chan2.mean()
    # Compute std.
    chan0_std = chan0.std()
    chan1_std = chan1.std()
    chan2_std = chan2.std()
    print("Chan0 mean is:", chan0_mean)
    print("Chan1 mean is:", chan1_mean)
    print("Chan2 mean is:", chan2_mean)
    print("Chan0 std is:", chan0_std)
    print("Chan1 std is:", chan1_std)
    print("Chan2 std is:", chan2_std)

# Normalized correctly