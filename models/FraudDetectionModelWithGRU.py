class FraudDetectionModelWithGRU(nn.Module):
    def __init__(self, bert_model, hidden_size, gru_hidden_size, num_classes=2):
        super(FraudDetectionModelWithGRU, self).__init__()
        self.bert = bert_model

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(gru_hidden_size * 2, num_classes)

        for param in self.bert.bert.parameters():
            param.requires_grad = False

        for param in self.bert.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]  # last layer (batch_size, seq_len, hidden_size)

        gru_output, _ = self.gru(last_hidden_state)  # gru_output: (batch_size, seq_len, gru_hidden_size * 2)

        # Using [CLS] token
        cls_representation = gru_output[:, 0, :]  # (batch_size, gru_hidden_size * 2)

        # classifier
        logits = self.classifier(cls_representation)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits
