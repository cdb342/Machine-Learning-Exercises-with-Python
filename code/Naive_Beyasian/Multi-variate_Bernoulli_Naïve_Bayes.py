import numpy as np
def load_dataset():
    training_text=[['Chinese','Beijing','Chinese'],['Chinese','Chinese','Shanghai'],['Chinese','Macao'],['Tokyo','Japan','Chinese']]
    training_label=['C','C','C','J']
    test_text=[['Chinese','Chinese','Chinese','Tokyo','Japan'],['Tokyo','Tokyo','Japan','Shanghai']]
    return training_text,training_label,test_text
class Multinomial_Naive_Bayesian():
    def __init__(self,training_text,training_label):
        self.training_text=training_text
        self.training_label=training_label
    def creat_vocabulary_set(self):
        vocabulary_set=set()
        for text in self.training_text:
            vocabulary_set=vocabulary_set|set(text)#将训练集中的每个text转换成集合形式，再取集合的并集
        self.vocabulary_set=list(vocabulary_set)
    def creat_label_set(self):
        label_set=set(self.training_label)#将训练集标签转换成集合形式，即可去除重复项
        self.label_set= list(label_set)
    def bag_of_words(self,input_text):
        text_vectors = []
        for text in input_text:
            text_vector = [0] * len(self.vocabulary_set)
            for voc in text:
                if voc in self.vocabulary_set:
                    text_vector[self.vocabulary_set.index(voc)] = 1
            text_vectors.append(text_vector)
        return  text_vectors
    def pior_probability(self):
        pior_pro = [0] * len(self.label_set)
        for label in self.label_set:
            for l in self.training_label:
                if l == label:
                    pior_pro[self.label_set.index(label)] += 1
            pior_pro[self.label_set.index(label)] = pior_pro[self.label_set.index(label)] / len(training_label)
        return pior_pro
    def conditional_probability(self):
        pior_pro=self.pior_probability()
        training_text_vector=self.bag_of_words(self.training_text)
        cond_pro=[0] * len(self.label_set)
        for label in self.label_set:
            cond_pro[self.label_set.index(label)] = np.sum(np.array(training_text_vector)[[k for k in range(len(self.training_label)) if self.training_label[k] == label]],axis=0)
            cond_pro[self.label_set.index(label)] = (1+cond_pro[self.label_set.index(label)])/(2+pior_pro[self.label_set.index(label)]*len(self.training_label))
        return  cond_pro
    def predict(self,text):
        self.creat_vocabulary_set()
        self.creat_label_set()
        print('label_set:',self.label_set)
        pior_pro = self.pior_probability()
        print('pior_pro:',pior_pro)
        cond_pro=self.conditional_probability()
        print('cond_pro:',cond_pro)
        text_vector=self.bag_of_words(text)
        print('test_text_vector:',text_vector)
        result=[0]*len(text)
        tem = [0] * len(self.label_set)
        for i in range(len(result)):
            for j in range(len(self.label_set)):
                tem[j] = (cond_pro[j] ** np.array(text_vector)[i])*(1-cond_pro[j])**(1-np.array(text_vector)[i])
                tem[j] = pior_pro[j] * np.prod(tem[j])
            result[i]=self.label_set[tem.index(max(tem))]
            print(tem)
        return result
if __name__ == '__main__':
    training_text,training_label,test_text=load_dataset()
    aa=Multinomial_Naive_Bayesian(training_text,training_label)
    result=aa.predict(test_text)
    print('test text predict class:',result)
