import torch
from torch.utils.data import Dataset


class RREDDataset(Dataset):

    def __init__(self, sent1_list, sent2_list, labels=None):
        self.sent1_list = sent1_list
        self.sent2_list = sent2_list
        self.labels = labels if labels is not None else None
    
        assert len(self.sent1_list) == len(self.sent2_list)

    def __len__(self):
        return len(self.sent1_list)
    
    def __getitem__(self, idx):
        sent1 = str(self.sent1_list[idx])
        sent2 = str(self.sent2_list[idx])
        label = self.labels[idx] if self.labels is not None else None

        return {
            'sent1': sent1,
            'sent2': sent2,
            'label': label
        }

        
class TokenizerWrapper():

    def __init__(
        self,
        tokenizer,
        max_length,
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.CLS, self.CLS_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.SEP, self.SEP_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.PAD, self.PAD_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id

    def collate(self, samples):

        sent1_list = [sample['sent1'] for sample in samples]
        sent2_list = [sample['sent2'] for sample in samples]

        if samples[0]['label'] is not None:
            labels = [sample['label'] for sample in samples]
        else:
            labels = None

        encoding = self.tokenizer(
            text=sent1_list, 
            text_pair=sent2_list, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        return {
            'input_ids': torch.LongTensor(encoding['input_ids']),
            'attention_mask': torch.LongTensor(encoding['attention_mask']),
            'labels': torch.LongTensor(labels) if labels is not None else None,
        }

    def customized_collate(self, samples):
        '''
        this function is needed to modify collate in detail,
        In current funciton, the length of each sentence is set as half of max_length 
        '''
        
        sent1_list = [sample['sent1'] for sample in samples]
        sent2_list = [sample['sent2'] for sample in samples]

        if samples[0]['label'] is not None:
            labels = [sample['label'] for sample in samples]
        else:
            labels = None

        inputs = self.preprocess(sent1_list, sent2_list)

        # Convert tokens to token_ids
        input_ids = [self.tokenizer.convert_tokens_to_ids(text)for text in inputs]

        attention_mask = [
            [float(token != self.PAD_id) for token in text]
            for text in input_ids
        ]

        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(labels) if labels is not None else None,
        }

    def preprocess(self, sent1_list, sent2_list):
        
        inputs = []
        for i in range(len(sent1_list)):
            if self.max_length % 2 == 1:
                self.sent1_length = (self.max_length // 2) + 1
            else:
                self.sent1_length = self.max_length // 2

            sent1 = []
            sent1 += [self.CLS]
            sent1 += self.tokenizer.tokenize(sent1_list[i])
            sent1 += [self.SEP]
            if len(sent1) > self.sent1_length:
                sent1 = sent1[:self.sent1_length-1] + [self.SEP]

            sent2 = []
            sent2 += self.tokenizer.tokenize(sent2_list[i])
            sent2 += [self.SEP]

            text = sent1 + sent2
            if len(text) < self.max_length:
                text += [self.PAD]*(self.max_length-len(text))
            else:
                text = text[:self.max_length-1] + [self.SEP]

            inputs.append(text)

        return inputs