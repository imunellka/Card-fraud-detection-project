import torch
import torch.nn as nn


class FraudDetectionModelWithLSTMAttention(nn.Module):
    def __init__(self, bert_model, hidden_size, lstm_hidden_size, num_classes=2):
        super(FraudDetectionModelWithLSTMAttention, self).__init__()
        self.bert = bert_model

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(lstm_hidden_size * 2, 1)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_classes)

        for param in self.bert.bert.parameters():
            param.requires_grad = False

        for param in self.bert.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]  # las layer (batch_size, seq_len, hidden_size)

        lstm_output, _ = self.lstm(last_hidden_state)  # lstm_output: (batch_size, seq_len, lstm_hidden_size * 2)

        # Attention
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, lstm_hidden_size * 2)

        # Classifier
        logits = self.classifier(context_vector)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits
