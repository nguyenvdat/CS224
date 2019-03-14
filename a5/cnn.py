#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, e_char, k, f):
    	# Add convolution layer to combine character embeddings in to a word embedding 
    	# @param e_char: size of character embedding
    	# @param k: kernel size
    	# f: filter size which is size of word embedding
        super().__init__()
        self.conv1d = nn.Conv1d(e_char, f, k)
        nn.init.kaiming_normal_(self.conv1d.weight)

    def forward(self, x_reshaped):
        # Add connection for CNN
        # @param x_reshaped: embedding of character in batch of words (word_batch_size, e_char, m_word) (m_word = max length of word)
        # @returns batch of word embedding (word_batch_size, e_word)
        x_conv = self.conv1d(x_reshaped)
        x_conv_out = torch.max(F.relu(x_conv), -1)[0]
        return x_conv_out

def test_CNN():
    x = torch.tensor([[[1, 0], [1, 3], [6, 2], [9, 3], [1, 5]], [[2, 1], [0, 2], [3, 6], [4, 8], [2, 5]]], dtype=torch.float32)
    x_reshaped = x.transpose(1, 2)
    model = CNN(2, 3, 4)
    model.conv1d.weight.data = torch.tensor([[[2, 3, 1], [1, 5, 3]], [[5, 8, 6], [9, 3, 5]], [[5, 8, 6], [9, 3, 5]], [[5, 8, 6], [9, 3, 5]]], dtype=torch.float32)
    model.conv1d.bias.data = torch.tensor([1, 2, 4, 2], dtype=torch.float32)
    x_conv_out = model(x_reshaped)
    x_conv_out_gold = torch.tensor([[ 73., 162., 164., 162.],
       [ 82., 164., 166., 164.]])
    assert x_conv_out.size() == x_conv_out_gold.size(), "Shape is not correct: it should be {}".format(x_conv_out_gold.size())
    assert torch.mean(torch.abs(x_conv_out - x_conv_out_gold)) < 0.0001, "Value is not correct: it should be {}".format(x_conv_out_gold)

test_CNN()


### END YOUR CODE

