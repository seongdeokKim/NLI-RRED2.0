from transformers import BertModel, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.file_utils import ModelOutput

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class RREDForNLI(nn.Module):
    def __init__(
            self,
            config,
            num_labels,
            pretrained_model_name=None,
            freeze_embeddings=False,
    ):
        super(RREDForNLI, self).__init__()
        self.config = config
        self.num_labels = num_labels
        
        if pretrained_model_name == 'microsoft/BiomedVLP-CXR-BERT-specialized':
            # self.bert = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True)
            from health_multimodal.text.model import CXRBertModel
            self.bert = CXRBertModel.from_pretrained(pretrained_model_name, revision="v1.1")
        elif pretrained_model_name == 'emilyalsentzer/Bio_ClinicalBERT':
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
        elif pretrained_model_name == 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext':
            self.bert = AutoModel.from_pretrained(pretrained_model_name)
        elif pretrained_model_name == 'bert-base-uncased':
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError('You must select pretrained_model_name')
        print(f"###### Initialize model's weight from [{pretrained_model_name}] ######")

        if freeze_embeddings:
            print('###### Freeze bert embeddings parameters ######')
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(config.classifier_dropout)

        self.pooler = RREDPooler(config)
        hidden_size = self.pooler.dense.in_features

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            return_dict=True
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        # |sequence_output| = (batch_size, max_sequence, hidden_size)

        pooled_output = self.pooler(sequence_output)
        pooled_output = self.dropout(pooled_output)
        # |pooled_output| = (batch_size, hidden_size)

        logits = self.classifier(pooled_output)
        # |logits| = (batch_size, num_labels)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return RREDOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RREDPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # pool the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class RREDOutput(ModelOutput):
    loss: torch.FloatTensor
    logits: torch.FloatTensor
    hidden_state: torch.FloatTensor
    attentions: torch.FloatTensor
