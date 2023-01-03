import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from helper.models.bert_for_nli import RREDForNLI
from helper.data_loader import RREDDataset, TokenizerWrapper
from collections import OrderedDict

# Block the warning message of tokenizer
import logging
logging.disable(logging.WARNING)

def define_argparser():
    '''
    Define argument parser to take inference using fine-tuned model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', default='/home/workspace/helper/checkpoints/bert-base-uncased.pth')

    p.add_argument('--predict_fn', default='/home/data/radnli-report-inference/1.0.0/radnli_test_v1.jsonl')
    p.add_argument('--output_fn', default='/home/workspace/prediction.txt')
    p.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--top_k', type=int, default=1)

    config = p.parse_args()

    return config


def read_json_file(file_path):
    import json

    json_data = [json.loads(l) for l in open(file_path)]

    labels = []
    sent1_list = []
    sent2_list = []
    for d in json_data:
        sent1 = d['sentence1'].strip()
        sent2 = d['sentence2'].strip()
        label = d['gold_label'].strip()

        if sent1 != '' and sent2 != '':
            labels += [label]
            sent1_list += [sent1]
            sent2_list += [sent2]

    return (labels, sent1_list, sent2_list)

def get_loader(
    train_config, config, tokenizer,
    sent1_list, sent2_list
    ):

    # Get dataloaders using given tokenizer as collate_fn.
    data_loader = DataLoader(
        RREDDataset(sent1_list, sent2_list),
        batch_size=config.batch_size,
        #shuffle=True,
        collate_fn=TokenizerWrapper(tokenizer,
                                    train_config.max_length).collate,
    )

    return data_loader

def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.device != 'cuda' else 'cuda'
    )

    train_config = saved_data['config']
    checkpoint = saved_data['bert']
    index_to_label = saved_data['index_to_label']

    labels, sent1_list, sent2_list = read_json_file(config.predict_fn)

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        loaded_tokenizer = saved_data['tokenizer']
        data_loader = get_loader(
            train_config, config, loaded_tokenizer,
            sent1_list, sent2_list
        )

        model = RREDForNLI(
            config=train_config,
            num_labels=len(index_to_label),
            pretrained_model_name=train_config.pretrained_model_name
        )

        device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        if torch.cuda.is_available() and config.device == "cuda":
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = model.to(device)
            model = nn.DataParallel(model)
            msg = model.load_state_dict(checkpoint, strict=False)
        else:
            model = model.to(device)
            msg = model.load_state_dict(checkpoint, strict=False)
        print(f"###### Load model's weight from [{config.model_fn}] ######")
        print(msg)

        # Don't forget turn-on evaluation mode.
        model.eval()

        y_hats = []
        for step, mini_batch in enumerate(tqdm(data_loader)):
            input_ids = mini_batch['input_ids']
            input_ids = input_ids.to(device)
            attention_mask = mini_batch['attention_mask']
            attention_mask = attention_mask.to(device)

            # Take feed-forward
            logits = model(input_ids,
                           attention_mask=attention_mask)[0]
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs['logits']

            # current_y_hats = np.argmax(logits, axis=-1)
            current_y_hats = F.softmax(logits, dim=-1)
            y_hats += [current_y_hats]

        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        _, indices = y_hats.cpu().topk(k=len(index_to_label))

        # for i in range(len(labels)):
        #     if i == 0:
        #         sys.stdout.write('{}\t{}\t{}\n'.format(
        #             'label',
        #             'gold_label',
        #             'sent1',
        #             'sent2',
        #         ))
        #     sys.stdout.write('{}\t{}\t{}\t{}\n'.format(
        #         ",".join([index_to_label.get(int(j)) for j in indices[i][:config.top_k]]),
        #         labels[i],
        #         sent1_list[i],
        #         sent2_list[i],
        #     ))
        with open(config.output_fn, 'w', encoding='utf-8') as fw:
            for i in range(len(labels)):
                fw.write('{}\t{}\t{}\t{}\n'.format(
                    ",".join([index_to_label.get(int(j)) for j in indices[i][:config.top_k]]),
                    labels[i],
                    sent1_list[i],
                    sent2_list[i],
                ))

if __name__ == '__main__':

    config = define_argparser()
    main(config)
