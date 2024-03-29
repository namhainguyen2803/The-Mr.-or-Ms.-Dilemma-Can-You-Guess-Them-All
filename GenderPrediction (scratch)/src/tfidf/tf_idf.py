"""
    Created by @namhainguyen2803 on 29/05/2023
"""

import numpy as np

class Compute_TF_IDF():
    def __init__(self, list_document, dictionary=None, max_count=None, min_count=0, normalize_tf=False, smooth=True,
                 normalize_tfidf=None):
        self.list_document = list_document
        self.max_count = max_count
        self.min_count = min_count
        self.normalize_tf = normalize_tf
        self.smooth = smooth
        self.normalize_tfidf = normalize_tfidf

        self.dictionary = dictionary if dictionary != None else self.create_dictionary()
        self.num_word = len(self.dictionary)
        self.num_document = len(self.list_document)
        assert self.num_word != 0, "There is not any word in dictionary, please see lower min_count"
        self.word_to_index = self.mapping_word_to_index()
        self.matrix_word_count = self.create_count_matrix()
        self.idf_score = self.compute_idf()

    # Query word given index based on dictionary
    def retrieve_word(self, index):
        if 0 <= index <= self.num_word - 1:
            return self.dictionary[index]
        else:
            return -1

    # Query index given word based on dictionary
    def retrieve_index(self, word):
        return self.word_to_index.get(word.lower(), -1)

    # Split document into list of words
    def word_extraction(self, document):
        split_word = document.split()
        return split_word

    def map_word_to_count(self):
        dict_word_count = dict()
        for i in range(len(self.list_document)):
            list_word = self.word_extraction(self.list_document[i].lower())
            for j in range(len(list_word)):
                dict_word_count[list_word[j]] = dict_word_count.get(list_word[j], 0) + 1
        return dict_word_count

    def create_dictionary(self):
        if self.max_count == None and self.min_count == None:
            set_word = set()
            for document in self.list_document:
                set_word = set_word.union(set(self.word_extraction(document.lower())))
        else:
            set_word = set()
            mapping_word_count = self.map_word_to_count()
            for document in self.list_document:
                list_word = self.word_extraction(document.lower())
                for word in list_word:
                    if self.min_count != None:
                        if mapping_word_count[word] < self.min_count:
                            continue
                    if self.max_count != None:
                        if mapping_word_count[word] > self.max_count:
                            continue
                    set_word.add(word)
        return sorted(list(set_word))

    def mapping_word_to_index(self):
        dict_encode = dict()
        for i in range(len(self.dictionary)):
            dict_encode[self.dictionary[i]] = i
        return dict_encode

    def create_count_matrix(self):
        mat = np.zeros((self.num_document, self.num_word))
        for i in range(len(self.list_document)):
            document = self.list_document[i].lower()
            list_word = self.word_extraction(document)
            for j in range(len(list_word)):
                ind = self.retrieve_index(list_word[j])
                mat[i, ind] += 1
        return mat

    def compute_tf(self):
        length_name = np.sum(self.matrix_word_count, axis=1)
        if self.normalize_tf == True:
            return self.matrix_word_count / np.reshape(length_name, (-1, 1))
        else:
            return self.matrix_word_count

    def compute_idf(self):
        tmp = np.copy(self.matrix_word_count)
        tmp[tmp != 0] = 1
        num_doc_having_word = np.sum(tmp, axis=0)
        if self.smooth == True:
            # smoothen and avoid 0 in idf
            num_doc_having_word = np.log((self.num_document + 1) / (num_doc_having_word + 1)) + 1
        else:
            # avoid 0 in idf
            num_doc_having_word = np.log(self.num_document / num_doc_having_word) + 1
        return np.reshape(num_doc_having_word, (1, self.num_word))

    def compute_tf_idf(self):
        tf = self.compute_tf()
        idf = self.compute_idf()
        tfidf = tf * idf
        if self.normalize_tfidf == None:
            return tfidf
        elif self.normalize_tfidf == "l2":
            sum_squares = np.reshape(np.diag(tfidf.dot(tfidf)), (1, -1))
            return tfidf / sum_squares
        elif self.normalize_tfidf == "l1":
            sum_row = np.reshape(np.sum(tfidf, axis=1), (1, -1))
            return tfidf / sum_row

    def create_count_matrix_for_test(self, list_doc):
        mat = np.zeros((len(list_doc), self.num_word))
        for i in range(len(list_doc)):
            document = list_doc[i].lower()
            list_word = self.word_extraction(document)
            for j in range(len(list_word)):
                ind = self.retrieve_index(list_word[j])
                if ind != -1:
                    mat[i, ind] += 1
        return mat

    def compute_tf_for_test(self, matrix_count_document):
        length_name = np.sum(matrix_count_document, axis=1)
        if self.normalize_tf == True:
            return matrix_count_document / np.reshape(length_name, (-1, 1))
        else:
            return matrix_count_document

    def compute_tf_idf_for_test(self, document):
        matrix = self.create_count_matrix_for_test(document)
        tf = self.compute_tf_for_test(matrix)
        idf = self.idf_score
        tfidf = tf * idf
        if self.normalize_tfidf == None:
            return tfidf
        elif self.normalize_tfidf == "l2":
            sum_squares = np.reshape(np.diag(tfidf.dot(tfidf)), (1, -1))
            return tfidf / sum_squares
        elif self.normalize_tfidf == "l1":
            sum_row = np.reshape(np.sum(tfidf, axis=1), (1, -1))
            return tfidf / sum_row

if __name__ == "__main__":
    list_name = ["Nguyễn Nam Hải", "Phạm Quang Tùng", "Đoàn Thế Vinh", "Nguyễn Bá Thiêm"]
    TF_IDF = Compute_TF_IDF(list_name)
    print(TF_IDF.dictionary)
    print(TF_IDF.compute_tf_idf())
    print(TF_IDF.compute_tf_idf_for_test(["Hello My name is Hải"]))