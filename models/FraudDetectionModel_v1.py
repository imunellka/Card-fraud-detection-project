import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding


class FraudDetectionModel(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes=2):
        super(FraudDetectionModel, self).__init__()
        self.bert = bert_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # freezed BERT
        for param in self.bert.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Obtaining hidden states from BERT
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        pooled_output = last_hidden_state[:, 0, :]  # Obtaining [CLS] tokens
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits
