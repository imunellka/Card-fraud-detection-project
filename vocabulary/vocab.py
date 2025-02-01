from collections import OrderedDict
import numpy as np

# Dictionary class that allows attribute-style access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Vocabulary:
    def __init__(self, target_column_name="Is Fraud?"):
        # Special tokens used in the vocabulary
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.mask_token = "[MASK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"

        self.target_column_name = target_column_name  # Target column for classification
        self.special_field_tag = "SPECIAL"  # Special category for special tokens

        self.special_tokens = [self.unk_token, self.sep_token, self.pad_token,
                               self.cls_token, self.mask_token, self.bos_token, self.eos_token]

        # Mappings between tokens and their corresponding IDs
        self.token2id = OrderedDict()
        self.id2token = OrderedDict()
        self.field_keys = OrderedDict()
        self.token2id[self.special_field_tag] = OrderedDict()

        for token in self.special_tokens:
            global_id = len(self.id2token)
            local_id = len(self.token2id[self.special_field_tag])

            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            self.id2token[global_id] = [token, self.special_field_tag, local_id]

    def set_id(self, token, field_name, return_local=False):
        """
        Assigns a unique global and local ID to a token within a given field.
        If the token already exists, returns its existing ID.
        """
        global_id, local_id = None, None

        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])

            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        return local_id if return_local else global_id

    def get_id(self, token, field_name="", special_token=False, return_local=False):
        """
        Retrieves the global or local ID of a given token within a field.
        Raises an exception if the token is not found.
        """
        if special_token:
            field_name = self.special_field_tag

        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]
        else:
            raise Exception(f"Token {token} not found in field: {field_name}")

        return local_id if return_local else global_id

    def set_field_keys(self, keys):
        """
        Initializes field keys with empty OrderedDicts.
        """
        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = None

        self.field_keys[self.special_field_tag] = None

    def get_field_ids(self, field_name, return_local=False):
        """
        Retrieves all IDs for a given field.
        """
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            raise Exception(f"Field naming {field_name} is incorrect.")

        selected_idx = 1 if return_local else 0
        return [ids[idx][selected_idx] for idx in ids]

    def get_from_global_ids(self, global_ids, what_to_get='local_ids'):
        """
        Converts global IDs to local IDs or token names.
        """
        device = global_ids.device

        def map_global_ids_to_local_ids(gid):
            return self.id2token[gid][2] if gid != -100 else -100

        def map_global_ids_to_tokens(gid):
            return f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'

        if what_to_get == 'local_ids':
            return global_ids.cpu().apply_(map_global_ids_to_local_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_global_ids_to_tokens)
            new_array_for_tokens = global_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Invalid value for 'what_to_get'")

    def get_field_keys(self, remove_target=True, ignore_special=False):
        """
        Returns a list of field keys, optionally removing the target column and special tokens.
        """
        keys = list(self.field_keys.keys())

        if remove_target and self.target_column_name in keys:
            keys.remove(self.target_column_name)
        if ignore_special:
            keys.remove(self.special_field_tag)
        return keys

    def get_special_tokens(self):
        """
        Returns a dictionary mapping special token names to their corresponding values.
        """
        special_tokens_map = {}
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token

        return AttrDict(special_tokens_map)

    def __len__(self):
        """
        Returns the total number of tokens in the vocabulary.
        """
        return len(self.id2token)

    def __str__(self):
        """
        Returns a string representation of the vocabulary.
        """
        return f'vocab: [{len(self)} tokens]  [field_keys={self.field_keys}]'
