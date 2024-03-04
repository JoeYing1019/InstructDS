import os
import sys
import time
import json
import torch
import random
import argparse
import operator
import progressbar
import numpy as np
import torch.nn as nn
from torch import cuda
from torch.optim import Adam
from operator import itemgetter
import torch.nn.functional as F
from sklearn.metrics import classification_report
import wandb
from help import fix_random_state, NormalMetric, ReferMetric, remove_checkpoints
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

warnings.filterwarnings("ignore")

special_tokens = ['<_PAD_>', '<eos_utt>', '<sos_usr>', '<eos_usr>', '<sos_sys>', '<eos_sys>', '<sos_context>', '<eos_context>',
                  '<sos_s>', '<eos_s>', '<sos_a>', '<eos_a>', '<sos_ans>', '<eos_ans>']


def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--data_prefix', type=str, help='the path where stores the data.', default='dataset/dailydialogue')
    # model configuration
    parser.add_argument('--model_name', type=str, help='the model name', default='flan-t5-base')
    parser.add_argument('--pretrained_path', type=str, help='Pretrained checkpoint path.', default='google/flan-t5-base')
    # training configuration
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_save', type=int, default=0)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_rate", default=0.0, type=float, help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8, help='Batch size for each gpu.')
    parser.add_argument("--number_of_gpu", type=int, default=1, help="Number of available GPUs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation step.")
    parser.add_argument("--save_path", type=str, help="directory to save the model parameters.", default='ckpt')
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument("--random_state", "-rs", type=int, default=0)
    parser.add_argument('--log_name', default='flan-t5-base', type=str, help='')
    parser.add_argument('--interact', default='act_sen', type=str, help='')
    parser.add_argument('--tf', help='teaching force on test', action='store_true', default=False)
    parser.add_argument('--ins', help='whether to use instruction', action='store_true', default=False)
    parser.add_argument('--toy', help='whether to use smaller training set', action='store_true', default=False)

    return parser.parse_args()



def evaluate(args, metric, tf=False, mode='dev'):
    print('-----------------------------------------')
    print('Start evaluation dev at global update step {}'.format(global_step))
    model.eval()
    if not metric:
        reference = ReferMetric(
            len(sen_labels), len(act_labels),
            sen_labels['positive'], sen_labels['negative']
        )
    else:
        reference = NormalMetric()

    with torch.no_grad():
        dev_batch_list = data.get_batches(args.number_of_gpu * args.batch_size_per_gpu, mode=mode)
        dev_batch_num_per_epoch = len(dev_batch_list)
        dev_p = progressbar.ProgressBar(dev_batch_num_per_epoch)
        print('Number of evaluation {} batches is {}'.format(mode, dev_batch_num_per_epoch))
        dev_p.start()
        dev_pred_text_list, dev_reference_text_list = [], []
        res_list = []
        for p_dev_idx in range(dev_batch_num_per_epoch):
            dev_p.update(p_dev_idx)
            one_dev_batch = dev_batch_list[p_dev_idx]
            dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels, utt_list = data.parse_batch_tensor(
                one_dev_batch)
            if cuda_available:
                dev_batch_src_tensor = dev_batch_src_tensor.to(device)
                dev_batch_src_mask = dev_batch_src_mask.to(device)
                dev_batch_input = dev_batch_input.to(device)
                dev_batch_labels = dev_batch_labels.to(device)
            
            if tf:
                loss, one_dev_prediction_text_list = model(dev_batch_src_tensor, dev_batch_src_mask, dev_batch_input, dev_batch_labels)
            
            else:
                if multi_gpu_training:
                    one_dev_prediction_text_list = model.module.batch_prediction(dev_batch_src_tensor,
                                                                                dev_batch_src_mask)
                else:
                    one_dev_prediction_text_list = model.batch_prediction(dev_batch_src_tensor, dev_batch_src_mask)
                
            
            dev_pred_text_list += one_dev_prediction_text_list

            if multi_gpu_training:
                dev_reference_text_list += model.module.parse_batch_text(dev_batch_input)
            else:
                one_ref = model.parse_batch_text(dev_batch_input)
                dev_reference_text_list += one_ref
            

            for utt, pred_, gold_ in zip(utt_list, one_dev_prediction_text_list, one_ref):
                res_list.append({"utt":utt, "pred:":pred_, "ref":gold_})

        dev_p.finish()

        assert len(dev_pred_text_list) == len(dev_reference_text_list)
        dev_pred_sen = []
        dev_reference_sen = []
        dev_pred_act = []
        dev_reference_act = []
        
        for eva_idx in range(len(dev_pred_text_list)):
            if args.interact == 'sen_act':
                # sen_act
                sen_reference, act_reference = dev_reference_text_list[eva_idx].strip().split()
                pred_list = dev_pred_text_list[eva_idx].strip().split()

                if len(pred_list) == 2 and pred_list[0] in sen_labels.keys() and pred_list[1] in act_labels.keys():
                    dev_pred_sen.append(sen_labels[pred_list[0]])
                    dev_reference_sen.append(sen_labels[sen_reference])
                    dev_pred_act.append(act_labels[pred_list[1]])
                    dev_reference_act.append(act_labels[act_reference])
                else:
                    print('Error pred label: {}'.format(dev_pred_text_list[eva_idx]))
                    dev_pred_sen.append(9)
                    dev_reference_sen.append(sen_labels[sen_reference])
                    dev_pred_act.append(9)
                    dev_reference_act.append(act_labels[act_reference])

            elif args.interact == 'act_sen':
                # act_sen
                act_reference, sen_reference = dev_reference_text_list[eva_idx].strip().split()
                pred_list = dev_pred_text_list[eva_idx].strip().split()
                
                if len(pred_list) == 2 and pred_list[1] in sen_labels.keys() and pred_list[0] in act_labels.keys():
                    dev_pred_sen.append(sen_labels[pred_list[1]])
                    dev_reference_sen.append(sen_labels[sen_reference])
                    dev_pred_act.append(act_labels[pred_list[0]])
                    dev_reference_act.append(act_labels[act_reference])
                else:
                    print('Error pred label: {}'.format(dev_pred_text_list[eva_idx]))
                    if args.data_prefix=='dataset/mastodon':
                        if sen_labels[sen_reference] == 0:
                            dev_pred_sen.append(1)
                        elif sen_labels[sen_reference] == 1:
                            dev_pred_sen.append(2)
                        elif sen_labels[sen_reference] == 2:
                            dev_pred_sen.append(0)
                    else:
                        dev_pred_sen.append(9)

                    dev_reference_sen.append(sen_labels[sen_reference])
                    dev_pred_act.append(9)
                    dev_reference_act.append(act_labels[act_reference])

            elif args.interact == 'sen':
                # sen
                sen_reference = dev_reference_text_list[eva_idx].strip()
                pred = dev_pred_text_list[eva_idx].strip()
                
                if pred in sen_labels.keys():
                    dev_pred_sen.append(sen_labels[pred])
                    dev_reference_sen.append(sen_labels[sen_reference])
                else:
                    print('Error pred label: {}'.format(dev_pred_text_list[eva_idx]))
                    dev_pred_sen.append(9)
                    dev_reference_sen.append(sen_labels[sen_reference])

            elif args.interact == 'act':
                # act
                act_reference = dev_reference_text_list[eva_idx].strip()
                pred = dev_pred_text_list[eva_idx].strip()
                
                if pred in act_labels.keys():
                    dev_pred_act.append(act_labels[pred])
                    dev_reference_act.append(act_labels[act_reference])
                else:
                    print('Error pred label: {}'.format(dev_pred_text_list[eva_idx]))
                    dev_pred_act.append(9)
                    dev_reference_act.append(act_labels[act_reference])
                    
            
            else:
                print("error interact")


        assert len(dev_pred_sen) == len(dev_reference_sen)
        assert len(dev_pred_act) == len(dev_reference_act)
        if args.interact == 'act':
            sen_f1, sen_r, sen_p = 0.0, 0.0, 0.0
            act_f1, act_r, act_p = reference.validate_act(dev_pred_act, dev_reference_act)
        elif args.interact == 'sen':
            sen_f1, sen_r, sen_p = reference.validate_emot(dev_pred_sen, dev_reference_sen)
            act_f1, act_r, act_p = 0.0, 0.0, 0.0
        else:
            sen_f1, sen_r, sen_p = reference.validate_emot(dev_pred_sen, dev_reference_sen)
            act_f1, act_r, act_p = reference.validate_act(dev_pred_act, dev_reference_act)

        return sen_f1, sen_r, sen_p, act_f1, act_r, act_p


import argparse

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print('Using single GPU training.')
    else:
        pass

    args = parse_config()
    device = torch.device('cuda')
    x = vars(args)
    # fix random seed
    fix_random_state(args.random_state)


    wandb.init(
        project="project_name",
        config=x,
        name=args.log_name,
    )
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    print('Start loading data...')
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_path)
    tokenizer.add_tokens(special_tokens)

    if args.data_prefix == "dataset/mastodon":
        metric = False
        sen_labels = {'negative': 0, 'positive': 1, 'neutral': 2}
        act_labels = {'inform': 0, 'request': 1, 'exclamation': 2, 'question': 3, 'answer': 4, 'feedback': 5, 'open': 6, 'suggest': 7, 'agreement': 8, 'thank': 9, 'explicit': 10, 'hi': 11, 'disagreement': 12, 'offer': 13, 'sympathy': 14}
        data_name = 'mastodon'
    elif args.data_prefix == "dataset/dailydialogue":
        metric = True
        sen_labels = {'neutral': 0, 'anger': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'sadness': 5, 'surprise': 6}
        act_labels = {'inform': 0, 'question': 1, 'directive': 2, 'commissive': 3}
        data_name = 'dailydialog'
    else: 
            print('error data name')

    from dataclass import mastodon

    print('args_ins: {}'.format(str(args.ins)))

    train_path, dev_path, test_path = args.data_prefix + '/train.json', args.data_prefix + '/dev.json', args.data_prefix + '/test.json'
    data = mastodon(tokenizer, train_path, dev_path, test_path, data_name, args.ins, args.interact, args.toy)
    print('Data Loaded.')
    

    print('Start loading model...')
    from modelling.T5Model import T5Gen_Model

    model = T5Gen_Model(args.pretrained_path, tokenizer, dropout=args.dropout)
    wandb.watch(model, log="all")

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model)  # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print('Model loaded')

    # organize optimizer
    overall_batch_size = args.number_of_gpu * args.batch_size_per_gpu * args.gradient_accumulation_steps
    t_total = data.train_num * args.num_train_epochs // overall_batch_size
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rate * t_total),
                                                    num_training_steps=t_total)
    else:
        raise Exception('Wrong Optimizer Name!!!')

    optimizer.zero_grad()

    print('tearching force: {}'.format(str(args.tf)))

    global_step = 0
    best_dev_sen_f1 = 0.
    best_dev_act_f1 = 0.


    for epoch in range(args.num_train_epochs):
        model.train()
        # --- training --- #
        print('-----------------------------------------')
        print('Start training at epoch %d' % epoch)
        train_iterator = data.build_iterator(batch_size=args.number_of_gpu * args.batch_size_per_gpu, mode='train')
        train_batch_num_per_epoch = int(data.train_num / (args.number_of_gpu * args.batch_size_per_gpu))
        p = progressbar.ProgressBar(train_batch_num_per_epoch)
        p.start()
        p_train_idx = 0
        epoch_step, train_loss = 0, 0.
        for _, train_batch in enumerate(train_iterator):
            p.update(p_train_idx)
            p_train_idx += 1
            one_train_input_batch, one_train_output_batch, _ = train_batch
            if len(one_train_input_batch) == 0 or len(one_train_output_batch) == 0: break
            train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels, utt_list = \
                data.parse_batch_tensor(train_batch)
            if cuda_available:
                train_batch_src_tensor = train_batch_src_tensor.to(device)
                train_batch_src_mask = train_batch_src_mask.to(device)
                train_batch_input = train_batch_input.to(device)
                train_batch_labels = train_batch_labels.to(device)
            loss, _ = model(train_batch_src_tensor, train_batch_src_mask, train_batch_input, train_batch_labels)
            loss = loss.mean()
            loss.backward()
            wandb.log({'train_loss': loss.item()})
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            epoch_step += 1
            if (epoch_step + 1) % args.gradient_accumulation_steps == 0 or (
                    epoch_step + 1) == train_batch_num_per_epoch:
                optimizer.step()
                if args.optimizer_name == 'adam':
                    scheduler.step()  # only update learning rate for adam optimizer
                optimizer.zero_grad()
                global_step += 1
        p.finish()
        train_loss = train_loss / train_batch_num_per_epoch
        print('At epoch {}, total update steps is {}, the total training loss is {}'.format(epoch, global_step,
                                                                                            train_loss))
        print('++++++++++++++++++++++++++++++++++++++++++')

        # evaluate the model after specific training epoch
        if data_name == 'mastodon':
            tmp_epoch = 10
        else:
            tmp_epoch = 30

        if epoch >= tmp_epoch:
            dev_sen_f1, dev_sen_r, dev_sen_p, dev_act_f1, dev_act_r, dev_act_p = evaluate(args,
                                                                                          metric,
                                                                                          args.tf,
                                                                                          mode='dev')
            
            wandb.log(
                {'dev_sentiment_f1': dev_sen_f1, 'dev_sentiment_precision': dev_sen_p,
                 'dev_sentiment_recall': dev_sen_r,
                 'dev_act_f1': dev_act_f1, 'dev_act_precision': dev_act_p, 'dev_act_recall': dev_act_r})
            
            
            
            if dev_sen_f1 > best_dev_sen_f1 or dev_act_f1 > best_dev_act_f1:
                test_sen_f1, test_sen_r, test_sen_p, test_act_f1, test_act_r, test_act_p = evaluate(
                args, metric, args.tf, mode='test')
                
                print("<Epoch {:4d}>test score: sentiment f1: {:.4f} (r: "
                    "{:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
                    ";".format(epoch, test_sen_f1, test_sen_r, test_sen_p, test_act_f1, test_act_r, test_act_p))

                wandb.log(
                {'test_sentiment_f1': test_sen_f1, 'test_sentiment_precision': test_sen_p,
                 'test_sentiment_recall': test_sen_r,
                 'test_act_f1': test_act_f1, 'test_act_precision': test_act_p, 'test_act_recall': test_act_r})

                if dev_sen_f1 > best_dev_sen_f1:
                    best_dev_sen_f1 = dev_sen_f1
                    print("<Epoch {:4d}>, Update (base on dev sent) test score: sentiment f1: {:.4f} (r: "
                          "{:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
                          ";".format(epoch, test_sen_f1, test_sen_r, test_sen_p, test_act_f1, test_act_r, test_act_p))

                if dev_act_f1 > best_dev_act_f1:
                    best_dev_act_f1 = dev_act_f1
                    print("<Epoch {:4d}>, Update (base on dev act) test score: sentiment f1: {:.4f} (r: "
                          "{:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
                          ";".format(epoch, test_sen_f1, test_sen_r, test_sen_p, test_act_f1, test_act_r, test_act_p))

                model_save_path = args.save_path + '/epoch_{}_test_sen_f_{}_test_act_f_{}'.format(epoch,
                                                                                                  round(test_sen_f1, 4),
                                                                                                  round(test_act_f1, 4))
                model_save_path = args.save_path + '/model.pt'
                import os

                if os.path.exists(model_save_path):
                    pass
                else:  
                    os.makedirs(model_save_path, exist_ok=True)
                if multi_gpu_training:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)
                print('Model saved.')
                remove_checkpoints(args)

            model.train()
            print('dev evaluation finished.')
            print('Resume training....')
            print('-----------------------------------------')
