import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from helper.models.bert_for_nli import RREDForNLI
from helper.trainer import Trainer
from helper.data_loader import RREDDataset, TokenizerWrapper
from collections import OrderedDict

# Block the warning message of tokenizer
import logging
logging.disable(logging.WARNING)

def define_argparser():

    p = argparse.ArgumentParser()

    # p.add_argument('--model_fn', default='./helper/checkpoints/bert-base-uncased.pth')
    # p.add_argument('--model_fn', default='./helper/checkpoints/Bio_ClinicalBERT.pth')
    # p.add_argument('--model_fn', default='./helper/checkpoints/CXR-BERT-specialized.pth')
    # p.add_argument('--model_fn', default='./helper/checkpoints/PubMedBERT.pth')
    p.add_argument('--model_fn', default='./helper/checkpoints/cpu_test_model_bin.pth')

    p.add_argument('--train_fn', default='/home/data/mednli/1.0.0/mli_train_v1.jsonl')
    p.add_argument('--valid_fn', default='/home/data/radnli-report-inference/1.0.0/radnli_dev_v1.jsonl')
    p.add_argument('--test_fn', default='/home/data/radnli-report-inference/1.0.0/radnli_dev_v1.jsonl')

    p.add_argument('--pretrained_model_name', type=str, default='bert-base-uncased', choices=['bert-base-uncased', 'emilyalsentzer/Bio_ClinicalBERT', 'microsoft/BiomedVLP-CXR-BERT-specialized', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'])
    # p.add_argument('--checkpoint_path', type=str, default=None, choices=['/home/workspace/helper/checkpoints_from_others/test_pytorch_model.bin'])
    p.add_argument('--checkpoint_path', type=str, default='/home/workspace/helper/checkpoints_from_others/test_checkpoint.bin', choices=['/home/workspace/helper/checkpoints_from_others/test_checkpoint.bin'])

    p.add_argument('--hidden_size', type=int, default=768)
    p.add_argument('--freeze_embeddings', type=bool, default=False)

    # p.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])

    p.add_argument('--batch_size', type=int, default=700)
    p.add_argument('--n_epochs', type=int, default=1)

    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--beta1', type=float, default=.9)
    p.add_argument('--beta2', type=float, default=.999)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--max_length', type=int, default=100)
    p.add_argument('--classifier_dropout', type=float, default=.1)

    config = p.parse_args()

    return config


def read_json_file(file_path):
    import json

    json_data = [json.loads(l) for l in open(file_path)]

    samples = []
    for d in json_data:
        sent1 = d['sentence1'].strip()
        sent2 = d['sentence2'].strip()
        label = d['gold_label'].strip()

        if sent1 != '' and sent2 != '':
            samples += [(label, sent1, sent2)]

    return samples

def get_loader(config, dataset_path, tokenizer, index_to_label=None):

    samples = read_json_file(dataset_path)

    labels = [s[0] for s in samples]
    sent1_list = [s[1] for s in samples]
    sent2_list = [s[2] for s in samples]

    # Generate label to index map.
    if index_to_label is not None:
        label_to_index = {
            label: index for index, label in index_to_label.items()
        }
    else:
        unique_labels = list(set(labels))
        label_to_index = {}
        index_to_label = {}
        for i, label in enumerate(unique_labels):
            label_to_index[label] = i
            index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels))

    # Get dataloaders using given tokenizer as collate_fn.
    data_loader = DataLoader(
        RREDDataset(sent1_list, sent2_list, labels),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer,
                                    config.max_length).collate,
    )

    return data_loader, index_to_label

def get_tokenizer(pretrained_model_name):
    if pretrained_model_name == 'microsoft/BiomedVLP-CXR-BERT-specialized':
        # tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name, trust_remote_code=True)
        from health_multimodal.text.model import CXRBertTokenizer
        tokenizer = CXRBertTokenizer.from_pretrained(config.pretrained_model_name, revision='v1.1')
    elif pretrained_model_name == 'emilyalsentzer/Bio_ClinicalBERT':
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    elif pretrained_model_name == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    elif pretrained_model_name == 'bert-base-uncased':
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    return tokenizer

def main(config):
    # Get pretrained tokenizer.
    tokenizer = get_tokenizer(config.pretrained_model_name)

    # Get dataloaders using tokenizer from untokenized corpus.
    # train_loader, valid_loader, index_to_label, test_loader = get_loaders(config, tokenizer)
    train_loader, index_to_label = get_loader(config, config.train_fn, tokenizer)
    valid_loader, _ = get_loader(config, config.valid_fn, tokenizer, index_to_label)
    test_loader, _ = get_loader(config, config.test_fn, tokenizer, index_to_label)

    print(
        '|train| =', len(train_loader) * config.batch_size,
        '|valid| =', len(valid_loader) * config.batch_size,
        '|test| =', len(test_loader) * config.batch_size,
    )

    print(index_to_label)

    # Get model
    model = RREDForNLI(
        config,
        num_labels=len(index_to_label),
        pretrained_model_name=config.pretrained_model_name,
        freeze_embeddings=config.freeze_embeddings
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
    if torch.cuda.is_available() and config.device == "cuda":
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device)
        if config.checkpoint_path is not None:
            checkpoint = torch.load(config.checkpoint_path)
            new_checkpoint = OrderedDict()
            for k, v in checkpoint.items():
                # add `module`
                name = 'module.' + k
                new_checkpoint[name] = v
            msg = model.load_state_dict(new_checkpoint, strict=False)
    else:
        model = model.to(device)
        if config.checkpoint_path is not None:
            checkpoint = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
            msg = model.load_state_dict(checkpoint, strict=False)

    if config.checkpoint_path is not None:
        print(f"###### Load model's weight from the checkpoint[{config.checkpoint_path}] ######")
        print(msg)

    if config.use_radam:
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)
    else:
        # optimizer = optim.Adam(model.parameters())

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.adam_epsilon
        )

    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )

    # Start train.
    trainer = Trainer(config)

    best_model = trainer.train(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        index_to_label,
        device,
    )

    trainer.test(
        best_model,
        test_loader,
        index_to_label,
        device,
    )

    torch.save({
        'bert': best_model.state_dict(),
        'config': config,
        'pretrained_model_name': config.pretrained_model_name,
        'vocab': tokenizer.get_vocab(),
        'index_to_label': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':

    config = define_argparser()
    main(config)
