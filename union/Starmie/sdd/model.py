import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}

class TableModel(nn.Module):
    """A baseline model for Table/Column matching"""

    def __init__(self, device='cuda', lm='roberta'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768
        self.fc = torch.nn.Linear(hidden_size, 2)
        # self.fc = torch.nn.Linear(hidden_size, 1)
        # self.cosine = nn.CosineSimilarity()
        # self.distance = nn.PairwiseDistance()

    def forward(self, x):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x (LongTensor): a batch of ID's of the left+right

        Returns:
            Tensor: binary prediction
        """
        x = x.to(self.device) # (batch_size, seq_len)

        # left+right
        enc_pair = self.bert(x)[0][:, 0, :] # (batch_size, emb_size)

        batch_size = len(x)
        # left and right
        enc = self.bert(x)[0][:, 0, :]

        # enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
        # enc1 = enc[:batch_size] # (batch_size, emb_size)
        # enc2 = enc[batch_size:] # (batch_size, emb_size)

        # fully connected
        return self.fc(enc)



def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwinsSimCLR(nn.Module):
    """Barlow Twins or SimCLR encoder for contrastive learning.
    """
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768

        # projector
        self.projector = nn.Linear(hidden_size, hp.projector)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)

        # a fully connected layer for fine tuning
        self.fc = nn.Linear(hidden_size * 2, 2)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().to(device)

        # cls token id
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id


    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return logits, labels

    def _extract_columns(self, x, z, cls_indices=None):
        """Helper function for extracting column vectors from LM outputs.
        """
        x_flat = x.view(-1)
        column_vectors = z.view((x_flat.shape[0], -1))

        if cls_indices is None:
            indices = [idx for idx, token_id in enumerate(x_flat) \
                if token_id == self.cls_token_id]
        else:
            indices = []
            seq_len = x.shape[-1]
            for rid in range(len(cls_indices)):
                indices += [idx + rid * seq_len for idx in cls_indices[rid]]

        return column_vectors[indices]


    def inference(self, x):
        """Apply the model on a serialized table.

        Args:
            x (LongTensor): a batch of serialized tables

        Returns:
            Tensor: the column vectors for all tables
        """
        x = x.to(self.device)
        z = self.bert(x)[0]
        z = self.projector(z) # optional
        return self._extract_columns(x, z)


    def forward(self, x_ori, x_aug, cls_indices, mode="simclr"):
        """Apply the model for contrastive learning.

        Args:
            x_ori (LongTensor): the first views of a batch of tables
            x_aug (LongTensor): the second views of a batch of tables
            cls_indices (tuple of List): the cls_token alignment
            mode (str, optional): the name of the contrastive learning algorithm

        Returns:
            Tensor: the loss
        """
        if mode in ["simclr", "barlow_twins"]:
            # pre-training
            # encode
            batch_size = len(x_ori)
            x_ori = x_ori.to(self.device) # original, (batch_size, seq_len)
            x_aug = x_aug.to(self.device) # augment, (batch_size, seq_len)

            # encode each table (all columns)
            x = torch.cat((x_ori, x_aug)) # (2*batch_size, seq_len)
            z = self.bert(x)[0] # (2*batch_size, seq_len, hidden_size)

            # assert that x_ori and x_aug have the same number of columns
            z_ori = z[:batch_size] # (batch_size, seq_len, hidden_size)
            z_aug = z[batch_size:] # (batch_size, seq_len, hidden_size)

            cls_ori, cls_aug = cls_indices

            z_ori = self._extract_columns(x_ori, z_ori, cls_ori) # (total_num_columns, hidden_size)
            z_aug = self._extract_columns(x_aug, z_aug, cls_aug) # (total_num_columns, hidden_size)
            assert z_ori.shape == z_aug.shape

            z = torch.cat((z_ori, z_aug))
            z = self.projector(z) # (2*total_num_columns, projector_size)

            if mode == "simclr":
                # simclr
                logits, labels = self.info_nce_loss(z, len(z) // 2, 2)
                loss = self.criterion(logits, labels)
                return loss
            elif mode == "barlow_twins":
                # barlow twins
                z1 = z[:len(z) // 2]
                z2 = z[len(z) // 2:]

                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))

                # use --scale-loss to multiply the loss by a constant factor
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
        elif mode == "finetune":
            pass
            # TODO
            # x1 = x1.to(self.device) # (batch_size, seq_len)
            # x2 = x2.to(self.device) # (batch_size, seq_len)
            # x12 = x12.to(self.device) # (batch_size, seq_len)
            # # left+right
            # enc_pair = self.projector(self.bert(x12)[0][:, 0, :]) # (batch_size, emb_size)
            # batch_size = len(x1)

            # # left and right
            # enc = self.projector(self.bert(torch.cat((x1, x2)))[0][:, 0, :])
            # #enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            # enc1 = enc[:batch_size] # (batch_size, emb_size)
            # enc2 = enc[batch_size:] # (batch_size, emb_size)

            # return self.fc(torch.cat((enc_pair, (enc1 - enc2).abs()), dim=1))
