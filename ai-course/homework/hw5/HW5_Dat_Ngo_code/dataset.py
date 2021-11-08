# dataset.py
import torch


class HousingDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        super(HousingDataset, self).__init__()

        self.inputs = inputs
        self.labels = labels
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]).float(),\
               torch.tensor(self.labels[index]).view(1).float()


class TitanicDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        super(TitanicDataset, self).__init__()

        self.inputs = inputs
        self.labels = labels
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]).float(),\
               torch.tensor(self.labels[index]).view(1).float()


class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        super(FashionDataset, self).__init__()

        self.images = images
        self.labels = labels
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # get image and reshape to [w, d, 3]
        image = torch.tensor(self.images[index])
        image = torch.reshape(image, shape=(1, 28, 28))

        # get label
        label = torch.tensor(self.labels[index])
        return image.float(), label
