# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import torch
from data_util import config
from data_util.data_process import *
import torch.optim as optim
from tqdm import tqdm, trange
from data_util.Metrics import Intent_Metrics, Slot_Metrics
from model.cm_net import Joint_model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Radam import RAdam, AdamW, PlainRAdam
from data_util import miulab

use_cuda = config.use_gpu and torch.cuda.is_available()

def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
    """
    Compute the accuracy based on the whole predictions of
    given sentence, including slot and intent.
    """
    total_count, correct_count = 0.0, 0.0
    for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

        if p_slot == r_slot and p_intent == r_intent:
            correct_count += 1.0
        total_count += 1.0

    return 1.0 * correct_count / total_count
def set_seed():
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not config.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def dev(model, dev_loader, idx2slot):

    model.eval()
    eval_loss_intent = 0
    eval_loss_slot = 0
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []
    for i, batch in enumerate(tqdm(dev_loader, desc="Evaluating")):
        inputs, char_lists, slot_labels, intent_labels, masks, = batch
        # inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        if torch.cuda.is_available():
            inputs, char_lists, masks, intent_labels, slot_labels = \
                inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
        logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
        loss_intent, loss_slot = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)

        pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
        pred_intents.extend(pred_intent.cpu().numpy().tolist())
        true_intents.extend(intent_labels.cpu().numpy().tolist())
        eval_loss_intent += loss_intent.item()
        eval_loss_slot += loss_slot.item()
        slot_labels = slot_labels.cpu().numpy().tolist()

        for i in range(len(pred_slot)):  # 遍历每个batch的pred_slot
            pred = []
            true = []
            for j in range(len(pred_slot[i])):  # 遍历每条pred_slot的token

                pred.append(idx2slot[pred_slot[i][j].item()])
                true.append(idx2slot[slot_labels[i][j]])
            pred_slots.append(pred[1:-1])
            true_slots.append(true[1:-1])
    # slot f1, p, r
    slot_f1, slot_p, slot_r = miulab.computeF1Score(true_slots, pred_slots)
    # intent f1, p, r
    Metrics_intent = Intent_Metrics(pred_intents, true_intents)
    intent_f1 = Metrics_intent.f1
    intent_acc = Metrics_intent.accuracy
    data_nums = len(dev_loader.dataset)
    ave_loss_intent = eval_loss_intent * config.batch_size / data_nums
    ave_loss_slot = eval_loss_slot * config.batch_size / data_nums

    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    print('\nEvaluation - intent_loss: {:.6f} slot_loss: {:.6f} acc: {:.4f}% '
          'intent f1: {:.4f} slot f1: {:.4f} sent acc: {:.4f} \n'.format(ave_loss_intent, ave_loss_slot,
                                                                         intent_acc, intent_f1,
                                                                         slot_f1, sent_acc))
    model.train()

    return intent_acc, slot_f1, sent_acc


def run_train(train_data_file, dev_data_file):

    print("1. load config and dict")
    vocab_file = open(config.data_path + "vocab.txt", "r", encoding="utf-8")
    vocab_list = [word.strip() for word in vocab_file]
    if not os.path.exists(config.data_path + "emb_word.txt"):
        emb_file = "D:/emb/glove.6B/glove.6B.300d.txt"
        embeddings = read_emb(emb_file, vocab_list)
        emb_write = open(config.data_path + "/emb_word.txt", "w", encoding="utf-8")
        for emb in embeddings:
            emb_write.write(emb)
        emb_write.close()
    else:
        embedding_file = open(config.data_path + "emb_word.txt", "r", encoding="utf-8")
        embeddings = [emb.strip() for emb in embedding_file]
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim)

    idx2intent, intent2idx = lord_label_dict(config.data_path + "intent_label.txt")
    idx2slot, slot2idx = lord_label_dict(config.data_path + "slot_label.txt")
    n_slot_tag = len(idx2slot.items())
    n_intent_class = len(idx2intent.items())

    train_dir = os.path.join(config.data_path, train_data_file)
    dev_dir = os.path.join(config.data_path, dev_data_file)
    train_loader = read_corpus(train_dir, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                               vocab=vocab, is_train=True)
    dev_loader = read_corpus(dev_dir, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                             vocab=vocab, is_train=False)
    model = Joint_model(config, config.hidden_dim, config.batch_size, config.max_len, n_intent_class, n_slot_tag, embedding_word)

    if use_cuda:
        model.cuda()
    model.train()
    optimizer = RAdam(model.parameters(), lr=config.lr, weight_decay=0.000001)
    # optimizer = getattr(optim,"Adam")
    # optimizer = optimizer(model.parameters(), lr = config.lr, weight_decay=0.00001)
    best_slot_f1 = [0.0, 0.0, 0.0]
    best_intent_acc = [0.0, 0.0, 0.0]
    best_sent_acc = [0.0, 0.0, 0.0]
    # best_slot_f1 = 0.0
    # best_intent_acc = 0.0
    # best_sent_acc = 0.0
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 60, 80], gamma=config.lr_scheduler_gama, last_epoch=-1)

    for epoch in trange(config.epoch, desc="Epoch"):
        print(scheduler.get_lr())
        step = 0
        for i, batch in enumerate(tqdm(train_loader, desc="batch_nums")):
            step += 1
            model.zero_grad()
            inputs, char_lists, slot_labels, intent_labels, masks, = batch
            # inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
            if use_cuda:
                inputs, char_lists, masks, intent_labels, slot_labels = \
                    inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
            logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
            loss_intent, loss_slot, = model.loss1(logits_intent, logits_slot, intent_labels, slot_labels, masks)

            if epoch < 40:
                loss = loss_slot + loss_intent
            else:
                loss = 0.8 * loss_intent + 0.2 * loss_slot
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("loss domain:", loss.item())
                print('epoch: {}|    step: {} |    loss: {}'.format(epoch, step, loss.item()))

        intent_acc, slot_f1, sent_acc = dev(model, dev_loader, idx2slot)
        # if slot_f1 > best_slot_f1 or intent_acc > best_intent_acc or sent_acc > best_sent_acc:
        #     torch.save(model, config.model_save_dir + config.model_name)
        if slot_f1 > best_slot_f1[1] :
            best_slot_f1 = [sent_acc, slot_f1, intent_acc, epoch]
            torch.save(model, config.model_save_dir + config.model_path)
        if intent_acc > best_intent_acc[2]:
            torch.save(model, config.model_save_dir + config.model_path)
            best_intent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        if sent_acc > best_sent_acc[0]:
            torch.save(model, config.model_save_dir + config.model_path)
            best_sent_acc = [sent_acc, slot_f1, intent_acc, epoch]
        scheduler.step()
    print("best_slot_f1:", best_slot_f1)
    print("best_intent_acc:", best_intent_acc)
    print("best_sent_acc:", best_sent_acc)
    # print("best_ave:", best_sent_ave)


def run_test(test_data_file):
    # load dict
    idx2intent, intent2idx = lord_label_dict(config.data_path + "intent_label.txt")
    idx2slot, slot2idx = lord_label_dict(config.data_path + "slot_label.txt")

    embedding_file = open(config.data_path + "emb_word.txt", "r", encoding="utf-8")
    embeddings = [emb.strip() for emb in embedding_file]
    embedding_word, vocab = process_emb(embeddings, emb_dim=config.emb_dim)

    test_dir = os.path.join(config.data_path, test_data_file)
    test_loader = read_corpus(test_dir, max_length=config.max_len, intent2idx=intent2idx, slot2idx=slot2idx,
                              vocab=vocab, is_train=False)
    model = torch.load(config.model_save_dir + config.model_path)
    if use_cuda:
        model.cuda()
    model.eval()
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        inputs, char_lists, slot_labels, intent_labels, masks, = batch
        # inputs, masks, tags = Variable(inputs), Variable(masks), Variable(tags)
        if torch.cuda.is_available():
            inputs, char_lists, masks, intent_labels, slot_labels = inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
        logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
        pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
        pred_intents.extend(pred_intent.cpu().numpy().tolist())
        true_intents.extend(intent_labels.cpu().numpy().tolist())
        #   intent_correct += (pred_intent.view(intent_labels.size()).data == intent_labels.data).sum().item()

        slot_labels = slot_labels.cpu().numpy().tolist()
        for i in range(len(pred_slot)):
            pred = []
            true = []
            for j in range(len(pred_slot[i])):
                pred.append(idx2slot[pred_slot[i][j].item()])
                true.append(idx2slot[slot_labels[i][j]])
            pred_slots.append(pred[1:-1])
            true_slots.append(true[1:-1])
    slot_f1 = miulab.computeF1Score(true_slots, pred_slots)[0]
    Metrics_intent = Intent_Metrics(pred_intents, true_intents)
    print(Metrics_intent.classification_report)
    intent_f1 = Metrics_intent.f1
    intent_acc = Metrics_intent.accuracy
    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    print('\nEvaluation -  acc: {:.4f}% ' 'intent f1: {:.4f} slot f1: {:.4f} sent_acc: {:.4f}  \n'.format(intent_acc,
                                                                                                        intent_f1,
                                                                                                        slot_f1,
                                                                                                        sent_acc))

    return intent_f1


if __name__ == "__main__":

    #
    train_file = "train.txt"
    dev_file = "dev.txt"
    test_file = "test.txt"
    #trian model
    set_seed()
    run_train(train_file, dev_file)
    #test model
    run_test(test_file)
