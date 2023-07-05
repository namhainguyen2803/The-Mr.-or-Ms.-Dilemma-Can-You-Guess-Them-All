import numpy as np

class TF_IDF():
  
  def __init__(self, corpus, dictionary=None, max_count=None, min_count=None, normalize_tf=False, smooth=True, normalize_tfidf=None):
    self.corpus=corpus
    self.max_count=max_count
    self.min_count=min_count
    self.normalize_tf=normalize_tf
    self.smooth=smooth
    self.normalize_tfidf=normalize_tfidf
    self.dictionary = dictionary if dictionary!=None else self.create_dictionary()
    self.num_word=len(self.dictionary)
    self.word_to_index = self.map_word_to_index()
    self.num_document = len(self.corpus)
    self.matrix_word_count = self.create_count_matrix()
  
  #return the word in dictionary given index
  def retrieve_word(self, index):
    return self.dictionary[index]
  
  def create_dictionary(self):
    if self.max_count==None and self.min_count==None:
      set_word = set()
      for doc in self.corpus:
        set_word = set_word.union(set(self.word_extraction(doc.lower())))
    else:
      set_word = set()
      map_word_count = self.map_word_to_count()
      for doc in self.corpus:
        list_word=self.word_extraction(doc.lower())
        for word in list_word:
          if self.min_count!=None:
            if map_word_count[word] < self.min_count:
              continue
          if self.max_count!=None:
            if map_word_count[word] > self.max_count:
              continue
          set_word.add(word)
    return sorted(list(set_word))
  
  def retrieve_index(self, word):
    return self.word_to_index[word.lower()]
  
  
  def word_extraction(self, document):
    split_word=document.split()
    return split_word

  def map_word_to_count(self):
    dict_word_count = dict()
    for i in range(len(self.corpus)):
      list_word = self.word_extraction(self.corpus[i].lower())
      for j in range(len(list_word)):
        dict_word_count[list_word[j]] = dict_word_count.get(list_word[j],0)+1
    return dict_word_count
  
  def map_word_to_index(self):
    dict_encode=dict()
    for i in range(len(self.dictionary)):
      dict_encode[self.dictionary[i]]=i
    return dict_encode
  
  def create_count_matrix(self):
    mat = np.zeros((self.num_document, self.num_word))
    for i in range(len(self.corpus)):
      document = self.corpus[i].lower()
      list_word = self.word_extraction(document)
      for j in range(len(list_word)):
        ind = self.retrieve_index(list_word[j])
        mat[i, ind]+=1
    return mat

  def compute_tf(self):
    length_name = np.sum(self.matrix_word_count, axis=1)
    if self.normalize_tf==True:
      return self.matrix_word_count/np.reshape(length_name, (-1,1))
    else:
      return self.matrix_word_count
  
  def compute_idf(self):
    tmp = np.copy(self.matrix_word_count)
    tmp[tmp!=0]=1
    num_doc_having_word = np.sum(tmp, axis=0)
    if self.smooth == True:
      num_doc_having_word = np.log((self.num_document+1) / (num_doc_having_word+1)) + 1
    else:
      num_doc_having_word = np.log(self.num_document / num_doc_having_word) + 1
    
    return np.reshape(num_doc_having_word, (1, self.num_word))
  def compute_tf_idf(self):
    tf = self.compute_tf()
    idf = self.compute_idf()
    tfidf = tf * idf
    if self.normalize_tfidf==None:
      return tfidf
    elif self.normalize_tfidf == "l2":
      sum_squares = np.reshape(np.diag(tfidf.dot(tfidf)), (1, -1))
      return tfidf / sum_squares
    elif self.normalize_tfidf == "l1":
      sum_row = np.reshape(np.sum(tfidf, axis=1), (1, -1))
      return tfidf / sum_row