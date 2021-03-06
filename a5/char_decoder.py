#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super().__init__() 
        pad_token_idx = target_vocab.char2id['<pad>']
        vocab_size = len(target_vocab.char2id)
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, vocab_size)
        self.decoderCharEmb = nn.Embedding(vocab_size, char_embedding_size, padding_idx=pad_token_idx)
        self.target_vocab = target_vocab

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_embedded = self.decoderCharEmb(input) # (length, batch, e_char)
        # output: (length, batch, hidden_size)
        # dec_hidden: tuple each of element has shape (1, batch, hidden_size)
        output, dec_hidden = self.charDecoder(input_embedded, dec_hidden)
        s_t = self.char_output_projection(output) # (length, batch, vocab_size)
        return s_t, dec_hidden

        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        length, batch = char_sequence.size()
        input_seq = char_sequence[0:-1]
        target_seq = char_sequence[1:]
        s_t, dec_hidden = self.forward(input_seq, dec_hidden)
        s_t = s_t.view(-1, len(self.target_vocab.char2id))
        target_seq = target_seq.contiguous().view(-1)
        mask = target_seq != self.target_vocab.char2id['<pad>']
        ce_loss = nn.CrossEntropyLoss(reduce=False)
        loss = ce_loss(s_t, target_seq)
        loss = loss * mask.type(torch.float)
        loss = torch.sum(loss) # zero out pad position
        return loss


        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        _, batch, hidden_size = initialStates[0].size()
        output_words = ['' for b in range(batch)]
        is_end = [False for b in range(batch)]
        current_char_index = self.target_vocab.start_of_word * torch.ones((1, batch), dtype=torch.long, device=device)
        prev_dec_hidden = initialStates
        for t in range(max_length):
            # s_t: (1, batch, vocab_size)
            # prev_dec_hidden: tuple each has shape (1, batch, hidden_size)
            s_t, prev_dec_hidden = self.forward(current_char_index, prev_dec_hidden)
            current_char_index = torch.argmax(s_t, dim=-1) # (1, batch)
            for b in range(batch):
                if current_char_index[0, b].item() == self.target_vocab.end_of_word:
                    is_end[b] = True
                if not is_end[b]:
                    output_words[b] += self.target_vocab.id2char[current_char_index[0, b].item()]
        return output_words


        
        ### END YOUR CODE

