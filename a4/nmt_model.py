#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import pdb

from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        
        # initialize hyperparameters
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # vocabulary
        self.vocab = vocab
        
        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = False
        self.counter = 0

        # embeddings
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)

        # encoder and decoder
        self.encoder = nn.LSTM( 
            input_size = embed_size,
            hidden_size = hidden_size,
            num_layers = 1,
            bias = True,
            bidirectional = True
        )                   

        self.decoder = nn.LSTMCell(
            input_size = embed_size + hidden_size,
            hidden_size = hidden_size,
            bias=True
        )

        # W_h, W_c, W_att, W_u, W_vocab
        self.h_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)  # W_h
        self.c_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)  # W_c
        self.att_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)  # W_att
        self.combined_output_projection = nn.Linear(hidden_size * 3, hidden_size, bias=False)  # W_u
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)  # W_vocab
        
        # dropout 
        self.dropout = nn.Dropout(dropout_rate) 

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]
        
        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (max_src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (max_tgt_len, b)

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths) 
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        
        # out = encode((embedding(source_padded))) # (src_len, b, vocab_size) => (b, src_len, 2h)
        out = self.model_embeddings.source(source_padded)  # (t, b, e)
        out = nn.utils.rnn.pack_padded_sequence(out, source_lengths)  
        out, (last_hidden, last_cell) = self.encoder(out)  # _, (b, 2h)
        out, len_unpacked = nn.utils.rnn.pad_packed_sequence(out)  # (t, b, 2h)
        out = out.transpose(0, 1)  # (b, t, 2h)

        # h_0_dec = W_h * [last_hidden[0], last_hidden[1]]
        # c_0_dec = W_c * [last_cell[0], last_cell[1]]
        h_0_dec = self.h_projection(torch.cat((last_hidden[0], last_hidden[1]), dim=1))  # (b, h)
        c_0_dec = self.c_projection(torch.cat((last_cell[0], last_cell[1]), dim=1))  # (b, h)

        return out, (h_0_dec, c_0_dec)
    

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state
    
        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step        
        ps = []
        
        # enc_hiddens_proj = W_attten * hi_enc
        enc_hiddens_proj = self.att_projection(enc_hiddens)  # (b, src_len, h)

        # ... 
        ys = self.model_embeddings.target(target_padded)     # (tgt_len, b, e)
        for y_t in torch.split(ys, 1):                   # (1, b, e)
            y_t = torch.squeeze(y_t, dim=0)                     # (b, e)
            y_hat_t = torch.cat((y_t, o_prev), dim=1)           # (b, e+h)
            dec_state, o_t, _ = self.step(y_hat_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            o_pred = o_t        # (b, h)
            ps.append(o_t)
            
        ps = torch.stack(ps, dim=0)  # (seq_len, b, h)
        ### END YOUR CODE

        return ps


    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:        
        """ Compute one forward step of the LSTM decoder, including the attention computation.        

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """
        # rename
        hs_enc = enc_hiddens        # (b, src_len, 2h)
        dec_state_pre = dec_state
        
        # dec_state_new = self.decoder(Ybar_t, dec_state
        dec_state_new = self.decoder(Ybar_t, dec_state_pre)  # (b, h)
        
        # e_t = ht^T_dec * (W_att * hi_enc)
        e_t = torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_state_new[0], dim=2)) # (b, src_len, 1)
        e_t = torch.squeeze(e_t, dim=2)  # (b, src_len)
    
        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        # compute a_t
        alpha_t = F.softmax(e_t, dim=1)  # (b, src_len)
        alpha_t = torch.unsqueeze(alpha_t, dim=1)  # (b, 1, src_len)
        a_t = torch.bmm(alpha_t, hs_enc)      # (b, 1, 2h)
        a_t = torch.squeeze(a_t, dim=1)            # (b, 2h)
        u_t = torch.cat((a_t, dec_state_new[0]), dim=1)  # (b, 3h)
        v_t = self.combined_output_projection(u_t)  # (b, h)
        O_t = self.dropout(torch.tanh(v_t))    # (b, h)

        return dec_state_new, O_t, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
        
