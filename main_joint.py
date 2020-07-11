# -*- coding: utf-8 -*-
import os
import random
import warnings

from data_util.data_process import *
from tqdm import tqdm, trange
from data_util.Metrics import IntentMetrics, SlotMetrics,semantic_acc
from model.joint_model_trans import Joint_model
from model.Radam import RAdam


warnings.filterwarnings('ignore')
if config.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda", torch.cuda.current_device())
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False


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
        if use_cuda:
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

        for i in range(len(pred_slot)):
            pred = []
            true = []
            for j in range(len(pred_slot[i])):
                pred.append(idx2slot[pred_slot[i][j].item()])
                true.append(idx2slot[slot_labels[i][j]])
            pred_slots.append(pred[1:-1])
            true_slots.append(true[1:-1])
    # slot f1, p, r
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, slot_p, slot_r = slot_metrics.get_slot_metrics()
    # intent f1, p, r
    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    intent_acc = Metrics_intent.accuracy
    data_nums = len(dev_loader.dataset)
    ave_loss_intent = eval_loss_intent * config.batch_size / data_nums
    ave_loss_slot = eval_loss_slot * config.batch_size / data_nums

    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    print('\nEvaluation - intent_loss: {:.6f} slot_loss: {:.6f} acc: {:.4f}% '
          'slot f1: {:.4f} sent acc: {:.4f} \n'.format(ave_loss_intent, ave_loss_slot,
                                                       intent_acc, slot_f1, sent_acc))
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
    best_slot_f1 = [0.0, 0.0, 0.0]
    best_intent_acc = [0.0, 0.0, 0.0]
    best_sent_acc = [0.0, 0.0, 0.0]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [40, 70], gamma=config.lr_scheduler_gama, last_epoch=-1)

    for epoch in trange(config.epoch, desc="Epoch"):
        print(scheduler.get_lr())
        step = 0
        for i, batch in enumerate(tqdm(train_loader, desc="batch_nums")):
            step += 1
            model.zero_grad()
            inputs, char_lists, slot_labels, intent_labels, masks, = batch
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
    model = torch.load(config.model_save_dir + config.model_path, map_location=device)
    model.eval()
    pred_intents = []
    true_intents = []
    pred_slots = []
    true_slots = []

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        inputs, char_lists, slot_labels, intent_labels, masks, = batch
        if use_cuda:
            inputs, char_lists, masks, intent_labels, slot_labels = inputs.cuda(), char_lists.cuda(), masks.cuda(), intent_labels.cuda(), slot_labels.cuda()
        logits_intent, logits_slot = model.forward_logit((inputs, char_lists), masks)
        pred_intent, pred_slot = model.pred_intent_slot(logits_intent, logits_slot, masks)
        pred_intents.extend(pred_intent.cpu().numpy().tolist())
        true_intents.extend(intent_labels.cpu().numpy().tolist())

        slot_labels = slot_labels.cpu().numpy().tolist()
        for i in range(len(pred_slot)):
            pred = []
            true = []
            for j in range(len(pred_slot[i])):
                pred.append(idx2slot[pred_slot[i][j].item()])
                true.append(idx2slot[slot_labels[i][j]])
            pred_slots.append(pred[1:-1])
            true_slots.append(true[1:-1])
    slot_metrics = SlotMetrics(true_slots, pred_slots)
    slot_f1, _, _ = slot_metrics.get_slot_metrics()

    Metrics_intent = IntentMetrics(pred_intents, true_intents)
    print(Metrics_intent.classification_report)
    intent_acc = Metrics_intent.accuracy
    sent_acc = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    print('\nEvaluation -  acc: {:.4f}% ' 'slot f1: {:.4f} sent_acc: {:.4f}  \n'.format(intent_acc, slot_f1, sent_acc))

    return sent_acc


if __name__ == "__main__":
    train_file = "train.txt"
    dev_file = "dev.txt"
    test_file = "test.txt"
    #trian model
    set_seed()
    run_train(train_file, dev_file)
    #test model
    run_test(test_file)
