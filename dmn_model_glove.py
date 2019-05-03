# 
# DMN Model Code
# Most of the code is from here: https://github.com/DSKSD/DeepNLP-models-Pytorch/blob/master/notebooks/10.Dynamic-Memory-Network-for-Question-Answering.ipynb
# slight modifications to include GloVe and modify hyperparameters
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from io import open

import pickle
import bcolz

import random
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict

from copy import deepcopy
import os
import re
import unicodedata
flatten = lambda l: [item for sublist in l for item in sublist]

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
random.seed(1024)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
word2index={'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
glove = {}
def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


def pad_to_batch(batch, w_to_ix): # for bAbI dataset
    fact,q,a = list(zip(*batch))
    max_fact = max([len(f) for f in fact])
    max_len = max([f.size(1) for f in flatten(fact)])
    max_q = max([qq.size(1) for qq in q])
    max_a = max([aa.size(1) for aa in a])
    
    facts, fact_masks, q_p, a_p = [], [], [], []
    for i in range(len(batch)):
        fact_p_t = []
        for j in range(len(fact[i])):
            if fact[i][j].size(1) < max_len:
                fact_p_t.append(torch.cat([fact[i][j], Variable(LongTensor([w_to_ix['<PAD>']] * (max_len - fact[i][j].size(1)))).view(1, -1)], 1))
            else:
                fact_p_t.append(fact[i][j])

        while len(fact_p_t) < max_fact:
            fact_p_t.append(Variable(LongTensor([w_to_ix['<PAD>']] * max_len)).view(1, -1))

        fact_p_t = torch.cat(fact_p_t)
        facts.append(fact_p_t)
        fact_masks.append(torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in fact_p_t]).view(fact_p_t.size(0), -1))

        if q[i].size(1) < max_q:
            q_p.append(torch.cat([q[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_q - q[i].size(1)))).view(1, -1)], 1))
        else:
            q_p.append(q[i])

        if a[i].size(1) < max_a:
            a_p.append(torch.cat([a[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_a - a[i].size(1)))).view(1, -1)], 1))
        else:
            a_p.append(a[i])

    questions = torch.cat(q_p)
    answers = torch.cat(a_p)
    question_masks = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in questions]).view(questions.size(0), -1)
    
    return facts, fact_masks, questions, question_masks, answers


def prepare_sequence(seq, to_index, use_glove=False):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def bAbI_data_load(path):
    try:
        data = open(path).readlines()
    except:
        print("Such a file does not exist at " + str(path))
        return None
    
    data = [d[:-1] for d in data]
    data_p = []
    fact = []
    qa = []
    try:
        for d in data:
            index = d.split(' ')[0]
            if index == '1':
                fact = []
                qa = []
            if '?' in d:
                temp = d.split('\t')
                q = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
                a = temp[1].split() + ['</s>']
                stemp = deepcopy(fact)
                data_p.append([stemp, q, a])
            else:
                tokens = d.replace('.', '').split(' ')[1:] + ['</s>']
                fact.append(tokens)
    except:
        print("Please check the data is right")
        return None
    return data_p


class DMN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embed_size, dropout_p=0.1):
        super(DMN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(input_size, embed_size, padding_idx=0) #sparse=True)
        self.input_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.question_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        
        self.gate = nn.Sequential(
                            nn.Linear(hidden_size * 4, hidden_size),
                            nn.Tanh(),
                            nn.Linear(hidden_size, 1),
                            nn.Sigmoid()
                        )
        
        self.attention_grucell =  nn.GRUCell(hidden_size, hidden_size)
        self.memory_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.answer_grucell = nn.GRUCell((hidden_size + embed_size), hidden_size)
        self.answer_fc = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_p)
        
    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(1, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden
    
    def init_weight(self):
        nn.init.xavier_uniform_(self.embed.state_dict()['weight'])
        
        for name, param in self.input_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)
        for name, param in self.question_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)
        for name, param in self.gate.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)
        for name, param in self.attention_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)
        for name, param in self.memory_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)
        for name, param in self.answer_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)
        
        nn.init.xavier_normal_(self.answer_fc.state_dict()['weight'])
        self.answer_fc.bias.data.fill_(0)
        
    def forward(self, facts, fact_masks, questions, question_masks, num_decode, episodes=3, is_training=False):
        """
        facts : (B,T_C,T_I) / LongTensor in List # batch_size, num_of_facts, length_of_each_fact(padded)
        fact_masks : (B,T_C,T_I) / ByteTensor in List # batch_size, num_of_facts, length_of_each_fact(padded)
        questions : (B,T_Q) / LongTensor # batch_size, question_length
        question_masks : (B,T_Q) / ByteTensor # batch_size, question_length
        """
        # Input Module
        C = [] # encoded facts
        for fact, fact_mask in zip(facts, fact_masks):
            embeds = self.embed(fact)
            if is_training:
                embeds = self.dropout(embeds)
            hidden = self.init_hidden(fact)
            outputs, hidden = self.input_gru(embeds, hidden)
            real_hidden = []

            for i, o in enumerate(outputs): # B,T,D
                real_length = fact_mask[i].data.tolist().count(0) 
                real_hidden.append(o[real_length - 1])

            C.append(torch.cat(real_hidden).view(fact.size(0), -1).unsqueeze(0))
        
        encoded_facts = torch.cat(C) # B,T_C,D
        
        # Question Module
        embeds = self.embed(questions)
        if is_training:
            embeds = self.dropout(embeds)
        hidden = self.init_hidden(questions)
        outputs, hidden = self.question_gru(embeds, hidden)
        
        if isinstance(question_masks, torch.autograd.Variable):
            real_question = []
            for i, o in enumerate(outputs): # B,T,D
                real_length = question_masks[i].data.tolist().count(0) 
                real_question.append(o[real_length - 1])
            encoded_question = torch.cat(real_question).view(questions.size(0), -1) # B,D
        else: # for inference mode
            encoded_question = hidden.squeeze(0) # B,D
            
        # Episodic Memory Module
        memory = encoded_question
        T_C = encoded_facts.size(1)
        B = encoded_facts.size(0)
        for i in range(episodes):
            hidden = self.init_hidden(encoded_facts.transpose(0, 1)[0]).squeeze(0) # B,D
            for t in range(T_C):
                z = torch.cat([
                                    encoded_facts.transpose(0, 1)[t] * encoded_question, # B,D , element-wise product
                                    encoded_facts.transpose(0, 1)[t] * memory, # B,D , element-wise product
                                    torch.abs(encoded_facts.transpose(0,1)[t] - encoded_question), # B,D
                                    torch.abs(encoded_facts.transpose(0,1)[t] - memory) # B,D
                                ], 1)
                g_t = self.gate(z) # B,1 scalar
                hidden = g_t * self.attention_grucell(encoded_facts.transpose(0, 1)[t], hidden) + (1 - g_t) * hidden
                
            e = hidden
            memory = self.memory_grucell(e, memory)
        
        # Answer Module
        answer_hidden = memory
        start_decode = Variable(LongTensor([[word2index['<s>']] * memory.size(0)])).transpose(0, 1)
        y_t_1 = self.embed(start_decode).squeeze(1) # B,D
        
        decodes = []
        for t in range(num_decode):
            answer_hidden = self.answer_grucell(torch.cat([y_t_1, encoded_question], 1), answer_hidden)
            decodes.append(self.answer_fc(answer_hidden))
        return torch.cat(decodes, 1).view(B * num_decode, -1)

def pad_to_fact(fact, x_to_ix): # this is for inference
    
    max_x = max([s.size(1) for s in fact])
    x_p = []
    for i in range(len(fact)):
        if fact[i].size(1) < max_x:
            x_p.append(torch.cat([fact[i], Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - fact[i].size(1)))).view(1, -1)], 1))
        else:
            x_p.append(fact[i])
        
    fact = torch.cat(x_p)
    fact_mask = torch.cat([Variable(ByteTensor(tuple(map(lambda s: s ==0, t.data))), volatile=False) for t in fact]).view(fact.size(0), -1)
    return fact, fact_mask

def get_accuracy(model, num_episode, test_data):
    accuracy = 0

    for t in test_data:
        fact, fact_mask = pad_to_fact(t[0], word2index)
        question = t[1]
        question_mask = Variable(ByteTensor([0] * t[1].size(1)), volatile=False).unsqueeze(0)
        answer = t[2].squeeze(0)
        
        model.zero_grad()
        pred = model([fact], [fact_mask], question, question_mask, answer.size(0), num_episode)
        if pred.max(1)[1].data.tolist() == answer.data.tolist():
            accuracy += 1

    return (accuracy/len(test_data) * 100)

#
# one training instance of the model itself
def DMNGlovemodelRun(train_path, test_path, babi_task_num, batch_size, epochs, use_glove, hiddensize, out_acc, ebsize=100):
    print("Starting for babi task " + babi_task_num)

    train_data = bAbI_data_load(train_path)

    print(train_data[0])

    fact,q,a = list(zip(*train_data))
    vocab = list(set(flatten(flatten(fact)) + flatten(q) + flatten(a)))

    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    index2word = {v:k for k, v in word2index.items()}

    for t in train_data:
        for i,fact in enumerate(t[0]):
            t[0][i] = prepare_sequence(fact, word2index, use_glove).view(1, -1)
        
        t[1] = prepare_sequence(t[1], word2index, use_glove).view(1, -1)
        t[2] = prepare_sequence(t[2], word2index, use_glove).view(1, -1)

    embed_size = ebsize
    GLOVE_PATH = "../../glove/glove.6B."+str(embed_size)+"d.txt"

    # INITIALIZE EMBEDDINGS TO RANDOM VALUES
    vocab_size = len(index2word)
    sd = 1/np.sqrt(embed_size)  # Standard deviation to use
    weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    weights = weights.astype(np.float32)

    # EXTRACT DESIRED GLOVE WORD VECTORS FROM TEXT FILE
    with open(GLOVE_PATH, encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            # Separate the values from the word
            line = line.split()
            word = line[0]

            # If word is in our vocab, then update the corresponding weights
            id = word2index.get(word, None)
            if id is not None:
                weights[id] = np.array(line[1:], dtype=np.float32)

    #TESTING
    test_data = bAbI_data_load(test_path)

    for t in test_data:
        for i, fact in enumerate(t[0]):
            t[0][i] = prepare_sequence(fact, word2index).view(1, -1)
        
        t[1] = prepare_sequence(t[1], word2index).view(1, -1)
        t[2] = prepare_sequence(t[2], word2index).view(1, -1)


    #TRAINING 
    HIDDEN_SIZE = hiddensize
    BATCH_SIZE = batch_size
    LR = 0.001
    EPOCH = epochs
    NUM_EPISODE = 3
    EARLY_STOPPING = False

    model = DMN(len(word2index), HIDDEN_SIZE, len(word2index), ebsize)
    model.embed.weight.data = torch.Tensor(weights)
    model.init_weight()
    if USE_CUDA:
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    accuracies = []

    for epoch in range(EPOCH):
        losses = []
        if EARLY_STOPPING: 
            break
            
        for i,batch in enumerate(getBatch(BATCH_SIZE, train_data)):
            facts, fact_masks, questions, question_masks, answers = pad_to_batch(batch, word2index)
            
            model.zero_grad()
            pred = model(facts, fact_masks, questions, question_masks, answers.size(1), NUM_EPISODE, True)
            loss = loss_function(pred, answers.view(-1))
            losses.append(loss.data.tolist())
            
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print("[%d/%d] mean_loss : %0.2f" %(epoch, EPOCH, np.mean(losses)))
                
                if np.mean(losses) < 0.001:
                    EARLY_STOPPING = False
                    print("Early Stopping!")
                    break
                losses = []
        
        if out_acc >= 1:
            acc = get_accuracy(model, NUM_EPISODE, test_data)
            accuracies.append(acc)
            print(acc)
    if out_acc >= 1:
        df = pd.DataFrame({'accuracy':accuracies})
        df.to_csv('results/q' + babi_task_num + '_epoch_accuracies.csv' )
    else:
        print(get_accuracy(model, NUM_EPISODE, test_data))
