from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tt

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from functools import partial

class TextDataset(Dataset):
    def __init__(self,
                csv_file: str,
                id_column: str,
                text_column: str,
                label_column: str,
                mode: str):
        """
        Args:
            csv_file (string): Path to the csv file with text and labels.
            text_column (string): Name of the column containing the text.
            label_column (string): Name of the column containing the labels.
        """
        data = pd.read_csv(csv_file)
        # Split the data
        train_data, val_data = train_test_split(data, train_size=0.7, random_state=42)
        if mode == 'train':
            self.data_frame = train_data
        elif mode == 'val':
            self.data_frame = val_data
        else:
            raise NotImplementedError
        
        self.id_column = id_column
        self.text_column = text_column
        self.label_column = label_column
        
        
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        id_ = self.data_frame.iloc[idx, self.data_frame.columns.get_loc(self.id_column)]
        text = self.data_frame.iloc[idx, self.data_frame.columns.get_loc(self.text_column)]
        label = self.data_frame.iloc[idx, self.data_frame.columns.get_loc(self.label_column)]
        
        sample = {'id': id_, 'text': text, 'label': label}
        return sample
    
    
    
def train_collate_fn(batch, tokenizer):
    texts = [sample['text'] for sample in batch]
    scores = [sample['label'] for sample in batch]
    enc_out = tokenizer(texts, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512,
                        padding=True)
    scores = torch.tensor(scores, dtype=torch.long) - 1 # map 1-6 to 0-5
    return dict(inps=enc_out, labels=scores)


def get_train_data(data_config):
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
    fn = partial(train_collate_fn, tokenizer=tokenizer)
    dataset = TextDataset(**data_config.dataset_config)
    data_loader = DataLoader(dataset=dataset, 
                                   **data_config.data_loader_config, 
                                   collate_fn=fn,
                                   shuffle=True)
    return dataset, data_loader
    
def get_val_data(data_config):
    tokenizer = AutoTokenizer.from_pretrained(data_config.tokenizer_name)
    fn = partial(train_collate_fn, tokenizer=tokenizer)
    dataset = TextDataset(**data_config.dataset_config)
    data_loader = DataLoader(dataset=dataset, 
                                   **data_config.data_loader_config, 
                                   collate_fn=fn,
                                   shuffle=False)
    return dataset, data_loader
