import sklearn.metrics as sk_metrics


class IntentMetrics(object):
    def __init__(self, intent_pred, intent_true):
        self.accuracy = sk_metrics.accuracy_score(intent_true, intent_pred)
        self.precision = sk_metrics.precision_score(intent_true, intent_pred, average="macro")
        self.recall = sk_metrics.recall_score(intent_true, intent_pred, average="macro")
        self.classification_report = sk_metrics.classification_report(intent_true, intent_pred)


class SlotMetrics(object):
    def __init__(self, correct_slots, pred_slots):
        self.correct_slots = correct_slots
        self.pred_slots = pred_slots

    def get_slot_metrics(self):
        correctChunk = {}
        correctChunkCnt = 0.0
        foundCorrect = {}
        foundCorrectCnt = 0.0
        foundPred = {}
        foundPredCnt = 0.0
        correctTags = 0.0
        tokenCount = 0.0
        for correct_slot, pred_slot in zip(self.correct_slots, self.pred_slots):
            inCorrect = False
            lastCorrectTag = 'O'
            lastCorrectType = ''
            lastPredTag = 'O'
            lastPredType = ''
            for c, p in zip(correct_slot, pred_slot):
                correctTag, correctType = SlotMetrics.splitTagType(c)
                predTag, predType = SlotMetrics.splitTagType(p)

                if inCorrect == True:
                    if SlotMetrics.endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                            SlotMetrics.endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                            (lastCorrectType == lastPredType):
                        inCorrect = False
                        correctChunkCnt += 1.0
                        if lastCorrectType in correctChunk:
                            correctChunk[lastCorrectType] += 1.0
                        else:
                            correctChunk[lastCorrectType] = 1.0
                    elif SlotMetrics.endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                            SlotMetrics.endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                            (correctType != predType):
                        inCorrect = False

                if SlotMetrics.startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                        SlotMetrics.startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (correctType == predType):
                    inCorrect = True

                if SlotMetrics.startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                    foundCorrectCnt += 1
                    if correctType in foundCorrect:
                        foundCorrect[correctType] += 1.0
                    else:
                        foundCorrect[correctType] = 1.0

                if SlotMetrics.startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                    foundPredCnt += 1.0
                    if predType in foundPred:
                        foundPred[predType] += 1.0
                    else:
                        foundPred[predType] = 1.0

                if correctTag == predTag and correctType == predType:
                    correctTags += 1.0

                tokenCount += 1.0

                lastCorrectTag = correctTag
                lastCorrectType = correctType
                lastPredTag = predTag
                lastPredType = predType

            if inCorrect == True:
                correctChunkCnt += 1.0
                if lastCorrectType in correctChunk:
                    correctChunk[lastCorrectType] += 1.0
                else:
                    correctChunk[lastCorrectType] = 1.0

        if foundPredCnt > 0:
            precision = 1.0 * correctChunkCnt / foundPredCnt
        else:
            precision = 0

        if foundCorrectCnt > 0:
            recall = 1.0 * correctChunkCnt / foundCorrectCnt
        else:
            recall = 0

        if (precision + recall) > 0:
            f1 = (2.0 * precision * recall) / (precision + recall)
        else:
            f1 = 0

        return f1, precision, recall

    @staticmethod
    def startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
        if prevTag == 'B' and tag == 'B':
            chunkStart = True
        if prevTag == 'I' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'B':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if prevTag == 'E' and tag == 'E':
            chunkStart = True
        if prevTag == 'E' and tag == 'I':
            chunkStart = True
        if prevTag == 'O' and tag == 'E':
            chunkStart = True
        if prevTag == 'O' and tag == 'I':
            chunkStart = True

        if tag != 'O' and tag != '.' and prevTagType != tagType:
            chunkStart = True
        return chunkStart

    @staticmethod
    def endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
        if prevTag == 'B' and tag == 'B':
            chunkEnd = True
        if prevTag == 'B' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'B':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag == 'E' and tag == 'E':
            chunkEnd = True
        if prevTag == 'E' and tag == 'I':
            chunkEnd = True
        if prevTag == 'E' and tag == 'O':
            chunkEnd = True
        if prevTag == 'I' and tag == 'O':
            chunkEnd = True

        if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
            chunkEnd = True
        return chunkEnd

    @staticmethod
    def splitTagType(tag):
        s = tag.split('-')
        if len(s) > 2 or len(s) == 0:
            raise ValueError('tag format wrong. it must be B-xxx.xxx')
        if len(s) == 1:
            tag = s[0]
            tagType = ""
        else:
            tag = s[0]
            tagType = s[1]
        return tag, tagType


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
