import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, AutoModelForSeq2SeqLM
import os

from learn import special_tokens


class T5Gen_Model(nn.Module):
    def __init__(self, model_path, tokenizer, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer 
        self.pad_token_id,self.eos_utt_token, self.sos_usr_token_id, self.eos_usr_token_id, self.sos_sys_token_id, self.eos_sys_token_id, self.sos_context_token_id, self.eos_context_token_id, self.sos_s_token_id, self.eos_s_token_id, self.sos_a_token_id, self.eos_a_token_id, self.sos_ans_token_id, self.eos_ans_token_id = self.tokenizer.convert_tokens_to_ids(
            special_tokens)

        print('Initializing Huggingface T5 model...')
        t5_config = T5Config.from_pretrained(model_path)
        t5_config.__dict__["dropout"] = dropout
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=t5_config, resume_download=True)
        print('Resizing Token Embeddings...')
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.tgt_sos_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_ans>'])[0]
        self.tgt_eos_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_ans>'])[0]

    def forward(self, src_input, src_mask, tgt_input, tgt_output):
        src_mask = src_mask.type(src_input.type())
        outputs = self.model(input_ids=src_input, attention_mask=src_mask, decoder_input_ids=tgt_input,
                             labels=tgt_output)
        loss = outputs[0]
        logits = outputs['logits']  
        text = self.parse_batch_text(torch.argmax(logits,dim=-1))
        return loss, text

    def parse_batch_text(self, batch_pred_ids):
        res_text_list = []
        for predicted_ids in batch_pred_ids:
            one_pred_ids = []
            for one_id in predicted_ids:
                if one_id in [self.pad_token_id,self.eos_utt_token, self.sos_usr_token_id, self.eos_usr_token_id, self.sos_sys_token_id, self.eos_sys_token_id, self.sos_context_token_id, self.eos_context_token_id, self.sos_s_token_id, self.eos_s_token_id, self.sos_a_token_id, self.eos_a_token_id, self.sos_ans_token_id, self.eos_ans_token_id]:
                    pass
                else:
                    one_pred_ids.append(one_id)
            one_res_text = self.tokenizer.decode(one_pred_ids)
            res_text_list.append(one_res_text)
        return res_text_list

    def batch_prediction(self, src_input, src_mask):
        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask,
                                      decoder_start_token_id=self.tgt_sos_token_id,
                                      pad_token_id=self.pad_token_id, eos_token_id=self.tgt_eos_token_id, max_length=64)
        return self.parse_batch_text(outputs)

    def save_model(self, ckpt_save_path):
        if not os.path.exists(ckpt_save_path):
            os.mkdir(ckpt_save_path)
        self.model.save_pretrained(ckpt_save_path)
        self.tokenizer.save_pretrained(ckpt_save_path)
