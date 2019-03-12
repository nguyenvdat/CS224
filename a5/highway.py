#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, e_word, dropout_rate):
    	# Set up the 2 linear layers for proj and gate and a dropout layer for the highway network
    	# @param e_word: size of the input x_conv_out and also size of the output x_word_emb
    	# @param dropout_rate: dropout rate for the drop out layer

        super().__init__()
        self.linear_proj = nn.Linear(e_word, e_word)
        nn.init.kaiming_normal_(self.linear_proj.weight)
        self.linear_gate = nn.Linear(e_word, e_word)
        nn.init.kaiming_normal_(self.linear_gate.weight)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Add connection for the highway network
        # @param x: x_conv_out final output of the Convolutional Network (batch_size, e_word)
        # @returns x_word_emb shape (batch_size, e_word)

        x_proj = F.relu(self.linear_proj(x))
        x_gate = torch.sigmoid(self.linear_gate(x))
        x_highway = x_gate * x_proj + (1 - x_gate) * x
        x_word_emb = self.dropout(x_highway)
        return x_word_emb

def test_Highway():
    x = torch.tensor([[1, 0], [2, 1]], dtype=torch.float32)
    model = Highway(2, 0)
    model.linear_proj.weight.data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    model.linear_proj.bias.data = torch.tensor([1, 2], dtype=torch.float32)
    model.linear_gate.weight.data = torch.tensor([[1, 0], [3, 0]], dtype=torch.float32)
    model.linear_gate.bias.data = torch.tensor([1, 0], dtype=torch.float32)
    x_word_emb = model(x)
    x_word_emb_gold = torch.tensor([[ 1.8808, 4.7629], [ 4.8577, 11.9728]])
    assert x_word_emb.size() == (2, 2), "Shape is not correct: it should be {}".format(x_word_emb_gold.size())
    assert torch.mean(torch.abs(x_word_emb - x_word_emb_gold)) < 0.0001, "Value is not correct: it should be {}".format(x_word_emb_gold)

test_Highway()

### END YOUR CODE 

