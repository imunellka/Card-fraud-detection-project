from transformers import BertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
)


class TabFormerBertConfig(BertConfig):
    def __init__(
        self,
        flatten=True,
        ncols=12,
        vocab_size=30522,
        field_hidden_size=64,
        hidden_size=768,
        num_attention_heads=12,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.ncols = ncols
        self.field_hidden_size = field_hidden_size
        self.hidden_size = hidden_size
        self.flatten = flatten
        self.vocab_size = vocab_size
        self.num_attention_heads=num_attention_heads



class ddict(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class TabFormerBertLM:
    def __init__(self, special_tokens, vocab, field_ce=False, flatten=True, ncols=None, field_hidden_size=768):
        self.ncols = ncols
        self.vocab = vocab
        vocab_file = "vocab_big_data"
        hidden_size = field_hidden_size

        self.config = TabFormerBertConfig(vocab_size=len(self.vocab),
                                          ncols=self.ncols,
                                          hidden_size=hidden_size,
                                          field_hidden_size=field_hidden_size,
                                          flatten=flatten,
                                          num_attention_heads=self.ncols)

        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False,
                                       **special_tokens)
        self.model = self.get_model(field_ce, flatten)

    def get_model(self, field_ce, flatten):
        return BertForMaskedLM(self.config)
