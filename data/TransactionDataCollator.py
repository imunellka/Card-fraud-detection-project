from typing import Dict, List, Tuple, Union
import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments


class TransDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        batch = self._tensorize_batch(examples)
        # batch = self._collate_batch(examples, self.tokenizer)
        sz = batch.shape
        batch = batch.view(sz[0], -1)
        inputs, labels = self.mask_tokens(batch)
        return {"input_ids": inputs.view(sz), "labels": labels.view(sz)}

    def _tensorize_batch(
         self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
     ) -> torch.Tensor:
         if isinstance(examples[0], (list, tuple)):
             examples = [torch.Tensor(e) for e in examples]
         length_of_first = examples[0].size(0)
         are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
         if are_tensors_same_length:
             return torch.stack(examples, dim=0)
         else:
             return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
