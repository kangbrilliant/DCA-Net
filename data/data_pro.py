vocab={}
intent_list=[]
slot_list=[]
intent_slot_list=[]
def read_data(path):
    f = open(path,"r",encoding="utf-8")
    sen_nums=0
    slot_labels = []
    for line in f:
        line = line.strip().split(" ")

        if len(line) == 2 :
            token, slot_label = line
            slot_labels.append(slot_label[2:])
            vocab[token] = vocab.get(token, 0) + 1
            if slot_label not in slot_list:
                slot_list.append(slot_label)
        if len(line)==1 and line!=[""]:


            intent = line[0]
            if intent == "atis_airfare#atis_flight":
                print(sen_nums)
            for slot_label_nob in slot_labels:
                if intent+"-"+slot_label_nob not in intent_slot_list:
                    intent_slot_list.append(intent+"-"+slot_label_nob)
            if intent not in intent_list:
                intent_list.append(intent)
            sen_nums = sen_nums+1
            slot_labels = []
if __name__ == '__main__':
    path = "./snips/"
    read_data(path+"train.txt")
    read_data(path+"test.txt")
    read_data(path+"dev.txt")
    intent_slot_list = sorted(intent_slot_list,key= lambda x :x )
    fw = open(path+"intent_slot_label.txt","w",encoding="utf-8")
    for i in intent_slot_list:
        fw.write(i+"\n")
    fw.close()
    print(intent_slot_list)
    # vocab_list = sorted([_ for _ in vocab.items() if _[1] >= 1], key=lambda x: x[1], reverse=True)
    #
    # vocab_list = [word_count[0] for word_count in vocab_list]
    # vocab_list = ["<pad>"]+["<unk>"]+["</s>"]+["</e>"]+vocab_list
    # fw = open(path + "vocab.txt","w",encoding="utf-8")
    # for line in vocab_list:
    #     fw.write(line+"\n")
    # fw.close()
    # intent_list.sort()
    #
    # slot_list = sorted([slot for slot in slot_list if len(slot) >1 or slot=="O"], key=lambda x: x)
    # slot_list = ["<PAD>"]+["<start>"]+["<end>"] + slot_list
    # f_intent = open(path + "intent_label.txt","w", encoding="utf-8")
    # f_slot = open(path+"slot_label.txt","w",encoding="utf-8")
    # for idx,line in enumerate(intent_list):
    #     f_intent.write(str(idx)+"\t"+line+"\n")
    # f_intent.close()
    # for idx,line in enumerate (slot_list):
    #     f_slot.write(str(idx)+"\t"+line+"\n")
    # f_slot.close()