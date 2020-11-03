import numpy as np

def load_dataset():
    training_text=[['Chinese','Beijing','Chinese'],['Chinese','Chinese','Shanghai'],['Chinese','Macao'],['Tokyo','Japan','Chinese']]
    training_label=['C','C','C','J']
    test_text=[['Chinese','Chinese','Chinese','Tokyo','Japan'],['Tokyo','Tokyo','Japan','Shanghai']]
class Multinomial_Naive_Bayesian():
    def __init__(self):
        pass
    def creat_vocabulary_set(self,training_text,training_label):
        vocabulary_set=set()
        for text in training_text:
            vocabulary_set=vocabulary_set|set(text)#将训练集中的每个text转换成集合形式，再取集合的并集
        return list(vocabulary_set)
    def creat_label_set(self,training_label):
        label_set=set(training_label)#将训练集标签转换成集合形式，即可去除重复项
        return list(label_set)
    def bag_of_words(self,training_text,vocabulary_set):
        training_text_vector = []
        for text in training_text:
            text_vector = [0] * len(vocabulary_set)
            for voc in text:
                if voc in vocabulary_set:
                    text_vector[vocabulary_set.index(voc)] += 1
            training_text_vector.append(text_vector)
        return  training_text_vector
    def pior_probability(self,training_label,label_set):
        pior_pro = [0] * len(label_set)
        for label in label_set:
            for l in training_label:
                if l == label:
                    pior_pro[label_set.index(label)] += 1
        return pior_pro
    def conditional_probability(self,label_set):
        cond_pro=[0] * len(label_set)
        for label in label_set:
            cond_pro[label_set.index(label)] = 1+np.sum(np.array(training_text_vector)[[k for k in range(len(training_label)) if training_label[k] == label]],axis=0)
            cond_pro[label_set.index(label)] = cond_pro[label_set.index(label)]/np.sum(cond_pro[label_set.index(label)])
        return  cond_pro
    def predict(self,text,label):
        result=[0]*len(label)
        tem = [0] * len(label_set)
        for i in range(len(result)):

            for j in len(label_set):
                tem[j] = cond_pro[j] ** np.array(text_vector)[j]
                tem[j] = pior_pro[j] * np.prod(tem)
                result=max(tem)
