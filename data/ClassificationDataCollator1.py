from typing import Dict, List, Union
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorWithPadding

class ClassificationDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, examples: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        labels = [example["label"] for example in examples]
        batch = self._tensorize_batch(input_ids)
        labels = torch.tensor(labels, dtype=torch.long)

        return {"input_ids": batch, "labels": labels}

    def _tensorize_batch(
        self, examples: List[torch.Tensor]
    ) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)
