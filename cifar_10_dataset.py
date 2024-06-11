from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tt


def get_train_data(data_config):
    def get_train_t(img_size, normalize_config):
        train_transform = tt.Compose([
            tt.Resize((img_size, img_size)),
            tt.RandomHorizontalFlip(p=0.5),
            tt.ToTensor(),
            tt.Normalize(normalize_config)
        ])
        return train_transform

    train_dataset = torchvision.datasets.CIFAR10(**data_config.dataset_config, 
                                            train=True, 
                                            transform=get_train_t(**data_config.transform_config), 
                                            target_transform=None, 
                                            download=False)
    train_data_loader = DataLoader(dataset=train_dataset, 
                                   **data_config.data_loader_config, 
                                   shuffle=True)
    return train_dataset, train_data_loader
    
def get_val_data(data_config):
    def get_val_t(img_size, normalize_config):
        val_transform = tt.Compose([
            tt.Resize((img_size, img_size)),
            tt.ToTensor(),
            tt.Normalize(normalize_config)
        ])
        return val_transform

    test_dataset = torchvision.datasets.CIFAR10(**data_config.dataset_config, 
                                            train=False, 
                                            transform=get_val_t(**data_config.transform_config), 
                                            target_transform=None, 
                                            download=False)
    test_data_loader = DataLoader(dataset=test_dataset, 
                                   **data_config.data_loader_config, 
                                   shuffle=False)
    return test_dataset, test_data_loader