import sklearn.metrics as sk_metrics


class Intent_Metrics(object):
    def __init__(self,intent_pred, intent_true):
        self.accuracy = sk_metrics.accuracy_score(intent_true, intent_pred)
        self.precision = sk_metrics.precision_score(intent_true, intent_pred ,average="macro")
        self.recall = sk_metrics.recall_score(intent_true, intent_pred ,average="macro")
        self.f1 = sk_metrics.f1_score(intent_true, intent_pred, average="macro")
        self.classification_report = sk_metrics.classification_report(intent_true ,intent_pred)


class Slot_Metrics(object):
    def __init__(self, golden_tags, predict_tags, label_list):

        # [[t1, t2], [t3, t4]...] --> [t1, t2, t3, t4...]
        golden_tags_without_end=[]
        # for i ,golden_tag in enumerate(golden_tags):
        #
        #     golden_tags_without_end.append(golden_tag[:-1])
        self.golden_tags = golden_tags
        self.predict_tags = predict_tags
        self.label_list = label_list

        (self.all_gold,self.all_right, self.all_pred), self.category_dict = self.count_entity(golden_tags, predict_tags, label_list)


        self.precision = self.precision_score(self.all_right,self.all_pred)
        self.recall = self.recall_score(self.all_right,self.all_gold)
        self.f1 = self.f1_score(self.precision,self.recall)

    def split_entity(self, label_sequence):

        entitys = {}
        entity_pointer = None
        for i, label in enumerate(label_sequence):
            if label.startswith('B'):
                category = label[2:]
                entity_pointer = (i, category)
            elif label.startswith("E"):
                if entity_pointer == None: continue
                elif entity_pointer[1] != label[2:]:continue
                entity_position = (entity_pointer[0], i)
                entitys [entity_position] = entity_pointer[1]
                entity_pointer = None
            elif label.startswith("S") and entity_pointer == None:
                entitys[i, i] = label[2:]
        return entitys

    def count_entity(self, golden_tags, predict_tags, label_list):

        golden_entitys = self.split_entity(golden_tags)
        pred_entitys = self.split_entity(predict_tags)
        all_gold = len(golden_entitys.items())
        all_pred = len(pred_entitys.items())

        category_dict = {}
        for label_category in label_list:
            entity_nums = EentityNums(0,0,0)

            category_dict[label_category] = entity_nums#entity_nums

        for pred_entity in pred_entitys.items():
            category_dict[pred_entity[1]].pred_nums +=1

        for golden_entity in golden_entitys.items():
            category_dict[golden_entity[1]].gold_nums +=1

        all_right = 0
        for golden_entity in golden_entitys.items():
            if golden_entity[0] in pred_entitys:
                if golden_entity[1] == pred_entitys[golden_entity[0]]:
                    category_dict[golden_entity[1]].right_nums += 1
                    all_right += 1
        return (all_gold, all_right, all_pred ), category_dict

    def f1_score(self, precision, recall):
        # precision = self.precision_score(right_nums, pred_nums)
        # recall = self.recall_score(right_nums, gold_nums)
        f1 = (2 * precision * recall) / (recall + precision) if (recall + precision) != 0 else 0.0
        return f1

    def precision_score(self, right_nums, pred_nums):
        return   right_nums/pred_nums if pred_nums!=0 else 0.0

    def recall_score(self, right_nums, gold_nums):
        return right_nums / gold_nums if gold_nums != 0 else 0.0

    def all_category_result(self):
        result=[['NO.', 'category','precision', 'recall', 'F1', 'right_nums',"gold_nums",'pred_nums']]
        for i, category in enumerate(self.category_dict.items()):
            precision = self.precision_score(category[1].right_nums, category[1].pred_nums)
            recall = self.precision_score(category[1].right_nums, category[1].gold_nums )
            f1 = self.f1_score(precision, recall)
            result.append([str(i), category[0], str(precision), str(recall), str(f1), str(category[1].right_nums), str(category[1].gold_nums), str(category[1].pred_nums)])
        return result


class EentityNums(object):
    def __init__(self,right_nums,pred_nums,gold_nums):
        self.right_nums = 0
        self.pred_nums = 0
        self.gold_nums = 0
