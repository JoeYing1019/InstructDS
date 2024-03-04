import codecs
import sys
import json
import torch
import random
import progressbar
import numpy as np
from torch.nn.utils import rnn


class mastodon:
    def __init__(self, tokenizer, train_path, dev_path, test_path, data_name, ins, interact, toy):
        self.tokenizer = tokenizer
        self.ins = ins
        self.toy = toy
        self.interact = interact
        self.data_name = data_name
        print('ins: {}'.format(str(self.ins)))
        print('toy: {}'.format(str(self.toy)))
        print('Interact: {}'.format(self.interact))
        print('Tokenizer size is {}'.format(len(tokenizer)))
        # dailydialog
        if data_name == 'dailydialog':
            self.sen_dict = { 0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happiness', 5: 'sadness', 6: 'surprise'}
            self.act_dict = {1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive'}
        # mastodon
        elif data_name == 'mastodon':
            self.sen_dict = {'-': 'negative', '+': 'positive', '*': 'neutral'}
            self.act_dict = {'I': 'inform', 'R': 'request', 'J': 'exclamation', 'Q': 'question', 'W': 'answer', 'F': 'feedback', 'O': 'open', 'S': 'suggest', 'A': 'agreement', 'T': 'thank', 'V': 'explicit', 'H': 'hi', 'D': 'disagreement', 'E': 'offer', 'M': 'sympathy'}
        else: 
            print('error data name')
        
        if self.interact == 'act':
            self.instruction = "Given a sentence within context, Please tell me the dialogue action from given dialogue action options."
            self.instruction += "Dialogue action options: " + ", ".join(self.act_dict.values()) + " \n" + "Text: " + "{}" + " \n" + "Answer:"
        elif self.interact == 'sen':
            self.instruction = "Given a sentence within context, Please tell me the sentiment from given sentiment options."
            self.instruction += "Sentiment options: " + ", ".join(self.sen_dict.values()) + " \n" + "Text: " + "{}" + " \n" + "Answer:"
        else:
            self.instruction = "Given a sentence within context, Please tell me the sentiment and dialogue action from given sentiment and dialogue action options."
            self.instruction += "Sentiment options: " + ", ".join(self.sen_dict.values()) + " \n" + "Dialogue action options: " + ", ".join(self.act_dict.values()) + " \n" + "Text: " + "{}" + " \n" + "Answer:"
        
        self.eos_utt = ' <eos_utt> '
        self.sos_ctx = ' <sos_context> '
        self.eos_ctx = ' <eos_context> '
        self.sen_tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_s>'])[0]
        self.sen_tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_s>'])[0]
        self.act_tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_a>'])[0]
        self.act_tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]

        self.tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_ans>'])[0]
        self.tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_ans>'])[0]

        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
        print('Loading training data...')
        self.train_data = self.load_path(train_path)
        if self.toy:
            toy_len = int(0.001 * len(self.train_data))
            self.train_data = self.train_data[:toy_len]
        self.train_data_id_list = self.reform_data(self.train_data)
        print('Training data size is {}'.format(len(self.train_data_id_list)))

        print('Loading Dev data...')
        self.dev_data = self.load_path(dev_path)
        self.dev_data_id_list = self.reform_data(self.dev_data)
        print('Dev data size is {}'.format(len(self.dev_data_id_list)))

        print('Loading test data...')
        self.test_data = self.load_path(test_path)
        self.test_data_id_list = self.reform_data(self.test_data)
        print('Test data size is {}'.format(len(self.test_data_id_list)))

        self.train_num, self.dev_num, self.test_num = len(self.train_data_id_list), len(self.dev_data_id_list), len(
            self.test_data_id_list)

    def get_utt_str(self, utts):
        his_str = ''
        for idx, utt in enumerate(utts[:-1]):
            his_str = his_str + utt + self.eos_utt

        his_str = self.sos_ctx + his_str + self.eos_ctx + utts[-1] + self.eos_utt

        return his_str

    def load_path(self, path):
        data = []
        with codecs.open(path, "r", "utf-8") as fr:
            dialogue_list = json.load(fr)
            for session in dialogue_list:
                utts = []
                for interact in session:
                    if self.data_name == 'dailydialog':
                        act = self.act_dict[int(interact["act"].strip('[').strip(']'))]
                        sentiment = self.sen_dict[int(interact["sentiment"].strip('[').strip(']'))]
                    elif self.data_name == 'mastodon':
                        act = self.act_dict[interact["act"].strip('[').strip(']')]
                        sentiment = self.sen_dict[interact["sentiment"].strip('[').strip(']')]
                    else:
                        print("error data name!")

                    utt = interact["utterance"]
                    utts.append(utt)
                    utt_str = self.get_utt_str(utts)
                    data.append({'utt_str': utt_str, 'sentiment': sentiment, 'act': act})
        return data

    def tokenize_text(self, text):
        token_id_list = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        token_id_list = [self.sos_context_token_id] + token_id_list + [self.eos_context_token_id]
        return token_id_list

    def tokenize_label(self, sen, act):
        sen_token_id_list = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sen))
        act_token_id_list = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(act))
        
        # act_sen
        if self.interact == 'sen_act':
            token_id_list = [self.tgt_sos_token_id] + sen_token_id_list + [self.sen_tgt_eos_token_id] + act_token_id_list + [self.act_tgt_eos_token_id] + [self.tgt_eos_token_id]
        elif self.interact == 'act_sen':
            token_id_list = [self.tgt_sos_token_id] + act_token_id_list + [self.act_tgt_eos_token_id] + sen_token_id_list + [self.sen_tgt_eos_token_id] + [self.tgt_eos_token_id]
        elif self.interact == 'act':
            token_id_list = [self.tgt_sos_token_id] + act_token_id_list + [self.act_tgt_eos_token_id] + [self.tgt_eos_token_id]
        elif self.interact == 'sen':
            token_id_list = [self.tgt_sos_token_id] + sen_token_id_list + [self.sen_tgt_eos_token_id] + [self.tgt_eos_token_id]
        else:
            print('Error Interaction')
        return token_id_list

    def reform_data(self, data):
        data_id_list = []
        sort_data = sorted(data, key=lambda x: len(x['utt_str']))
        for dic in sort_data:
            utt_str = dic['utt_str']
            if self.ins:
                utt_id_list = self.tokenize_text(self.instruction.format(utt_str))
            else:
                utt_id_list = self.tokenize_text(utt_str)
            
            sentiment = dic['sentiment']
            act = dic['act']
            label_id_list = self.tokenize_label(sentiment, act)
            data_id_list.append((utt_id_list, label_id_list, utt_str))

        return data_id_list

    def get_batches(self, batch_size, mode):
        batch_list = []
        if mode == 'train':
            all_data_list = self.train_data_id_list
        elif mode == 'dev':
            all_data_list = self.dev_data_id_list
        elif mode == 'test':
            all_data_list = self.test_data_id_list
        else:
            raise Exception('Wrong Mode!!!')

        all_input_data_list, all_output_data_list, all_utt_list = [], [], []
        for item in all_data_list:
            all_input_data_list.append(item[0])
            all_output_data_list.append(item[1])
            all_utt_list.append(item[2])

        data_num = len(all_input_data_list)
        batch_num = int(data_num / batch_size) + 1

        for i in range(batch_num):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            if start_idx > data_num - 1:
                break
            end_idx = min(end_idx, data_num - 1)
            one_input_batch_list, one_output_batch_list, one_utt_list = [], [], []
            for idx in range(start_idx, end_idx):
                one_input_batch_list.append(all_input_data_list[idx])
                one_output_batch_list.append(all_output_data_list[idx])
                one_utt_list.append(all_utt_list[idx])
            one_batch = [one_input_batch_list, one_output_batch_list, one_utt_list]
            if len(one_batch[0]) == 0:
                pass
            else:
                batch_list.append(one_batch)
        print('Number of {} batches is {}'.format(mode, len(batch_list)))
        return batch_list

    def build_iterator(self, batch_size, mode):
        batch_list = self.get_batches(batch_size, mode)
        for i, batch in enumerate(batch_list):
            yield batch

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list)
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list, one_utt_list = batch
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        batch_input, batch_labels = self.process_output(batch_output_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_src_tensor, batch_src_mask, batch_input, batch_labels, one_utt_list
    

