import os
from os import path
import pandas as pd
import numpy as np
import math
import tqdm
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data.dataset import Dataset

# Utility to divide a list into chunks of size `n`
def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Custom PyTorch Dataset for handling transaction data
class TransactionDataset(Dataset):
    def __init__(self,
                 user_ids=None,
                 seq_len=5,
                 num_bins=15,
                 root="./",
                 fname="card_transaction.v1.csv",
                 vocab_dir="checkpoints",
                 nrows=None,
                 flatten=True,
                 stride=3,
                 adap_thres=10 ** 8,
                 return_labels=False,
                 skip_user=False,
                 task_type="masking_learning"):
        # Initialize dataset parameters and configurations
        self.root = root
        self.fname = fname
        self.nrows = nrows
        self.user_ids = user_ids
        self.return_labels = return_labels
        self.skip_user = skip_user
        self.trans_stride = stride
        self.flatten = flatten
        self.vocab = Vocabulary(adap_thres)  # Vocabulary for encoding transaction fields
        self.seq_len = seq_len
        self.encoder_fit = {}  # To store encoders for different fields
        self.trans_table = None  # Transaction data
        self.data = []  # Encoded transaction data
        self.labels = []  # Transaction labels
        self.window_label = []  # Fraud labels for sliding windows
        self.ncols = None
        self.num_bins = num_bins
        self.task_type = task_type

        # Load and preprocess the data
        self.encode_data()
        # Initialize vocabulary with transaction field keys
        self.init_vocab()
        # Prepare the samples for training/testing
        self.prepare_samples()

    def __getitem__(self, index):
        return_data = torch.tensor(self.data[index], dtype=torch.long)

        if self.task_type == "classification":
            return{
                "input_ids": torch.tensor(self.data[index], dtype=torch.long),
                "label": self.window_label[index]
            }

        if self.return_labels:
            return_data = (return_data, torch.tensor(self.labels[index], dtype=torch.long))


        return return_data

    def __len__(self):
        return len(self.data)

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        # Fit and transform a column using LabelEncoder or MinMaxScaler
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)
        return mfit, mfit.transform(column)

    @staticmethod
    def timeEncoder(X):
        # Convert time columns into a single timestamp
        X_hm = X['Time'].str.split(':', expand=True)
        d = pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=X_hm[0], minute=X_hm[1])).astype(int)
        return pd.DataFrame(d)

    @staticmethod
    def amountEncoder(X):
        # Encode amount by applying logarithm transformation
        amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
        return pd.DataFrame(amt)

    @staticmethod
    def fraudEncoder(X):
        # Encode fraud column as binary (0/1)
        fraud = (X == 'Yes').astype(int)
        return pd.DataFrame(fraud)

    @staticmethod
    def nanNone(X):
        # Replace NaN values with 'None'
        return X.where(pd.notnull(X), 'None')

    @staticmethod
    def nanZero(X):
        # Replace NaN values with 0
        return X.where(pd.notnull(X), 0)

    def _quantization_binning(self, data):
        # Create bins for quantization of continuous values
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        # Quantize continuous data into discrete bins
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1
        return quant_inputs

    def user_level_data(self):
        # Aggregate data at the user level
        trans_data, trans_labels = [], []
        unique_users = self.trans_table["User"].unique()
        columns_names = list(self.trans_table.columns)

        for user in tqdm.tqdm(unique_users):
            user_data = self.trans_table.loc[self.trans_table["User"] == user]
            user_trans, user_labels = [], []
            for idx, row in user_data.iterrows():
                row = list(row)
                skip_idx = 1 if self.skip_user else 0
                user_trans.extend(row[skip_idx:-1])
                user_labels.append(row[-1])

            trans_data.append(user_trans)
            trans_labels.append(user_labels)

        if self.skip_user:
            columns_names.remove("User")

        return trans_data, trans_labels, columns_names

     def format_trans(self, trans_lst, column_names):
        # Format transactions into vocabulary IDs
        trans_lst = list(divide_chunks(trans_lst, len(self.vocab.field_keys) - 2))
        user_vocab_ids = []
        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for trans in trans_lst:
            vocab_ids = []
            for jdx, field in enumerate(trans):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)
            vocab_ids.append(sep_id)
            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids

      def prepare_samples(self):
        # Prepare input samples for training/testing
        trans_data, trans_labels, columns_names = self.user_level_data()

        for user_idx in tqdm.tqdm(range(len(trans_data))):
            user_row = trans_data[user_idx]
            user_row_ids = self.format_trans(user_row, columns_names)
            user_labels = trans_labels[user_idx]

            for jdx in range(0, len(user_row_ids) - self.seq_len + 1, self.trans_stride):
                ids = user_row_ids[jdx:(jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]
                self.data.append(ids)

            for jdx in range(0, len(user_labels) - self.seq_len + 1, self.trans_stride):
                ids = user_labels[jdx:(jdx + self.seq_len)]
                self.labels.append(ids)
                fraud = 0
                if len(np.nonzero(ids)[0]) > 0:
                    fraud = 1
                self.window_label.append(fraud)

        assert len(self.data) == len(self.labels)
        self.ncols = len(self.vocab.field_keys) - 2 + 1

      def get_csv(self, fname):
        # Load transaction data from a CSV file
        data = pd.read_csv(fname, nrows=self.nrows)
        if self.user_ids:
            self.user_ids = map(int, self.user_ids)
            data = data[data['User'].isin(self.user_ids)]

        self.nrows = data.shape[0]
        return data

      def init_vocab(self):
        # Initialize vocabulary with transaction data
        column_names = list(self.trans_table.columns)
        if self.skip_user:
            column_names.remove("User")

        self.vocab.set_field_keys(column_names)

        for column in column_names:
            unique_values = self.trans_table[column].value_counts(sort=True).to_dict()
            for val in unique_values:
                self.vocab.set_id(val, column)

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            if vocab_size > self.vocab.adap_thres:
                self.vocab.adap_sm_cols.add(column)
              
      def encode_data(self):
        # Preprocess and encode transaction data
        dirname = path.join(self.root, "preprocessed")
        data_file = path.join(self.root, self.fname)
        data = self.get_csv(data_file)

        # Handle missing values and encode fields
        data['Errors?'] = self.nanNone(data['Errors?'])
        data['Is Fraud?'] = self.fraudEncoder(data['Is Fraud?'])
        data['Zip'] = self.nanZero(data['Zip'])
        data['Merchant State'] = self.nanNone(data['Merchant State'])
        data['Use Chip'] = self.nanNone(data['Use Chip'])
        data['Amount'] = self.amountEncoder(data['Amount'])

        sub_columns = ['Errors?', 'MCC', 'Zip', 'Merchant State', 'Merchant City', 'Merchant Name', 'Use Chip']

        for col_name in tqdm.tqdm(sub_columns):
            col_data = data[col_name]
            col_fit, col_data = self.label_fit_transform(col_data)
            self.encoder_fit[col_name] = col_fit
            data[col_name] = col_data

        timestamp = self.timeEncoder(data[['Year', 'Month', 'Day', 'Time']])
        timestamp_fit, timestamp = self.label_fit_transform(timestamp, enc_type="time")
        self.encoder_fit['Timestamp'] = timestamp_fit
        data['Timestamp'] = timestamp

        coldata = np.array(data['Timestamp'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        data['Timestamp'] = self._quantize(coldata, bin_edges)
        self.encoder_fit["Timestamp-Quant"] = [bin_edges, bin_centers, bin_widths]

        coldata = np.array(data['Amount'])
        bin_edges, bin_centers, bin_widths = self._quantization_binning(coldata)
        data['Amount'] = self._quantize(coldata, bin_edges)
        self.encoder_fit["Amount-Quant"] = [bin_edges, bin_centers, bin_widths]
        
        columns_to_select = ['User',
                             'Card',
                             'Timestamp',
                             'Amount',
                             'Use Chip',
                             'Merchant Name',
                             'Merchant City',
                             'Merchant State',
                             'Zip',
                             'MCC',
                             'Errors?',
                             'Is Fraud?']

        self.trans_table = data[columns_to_select]
