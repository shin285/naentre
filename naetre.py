import numpy as np


def _convert_source_to_idx_(source, id_dic):
    if source not in id_dic:
        id_dic[source] = len(id_dic)
    return id_dic[source]


class Trainer:
    def __init__(self):
        self.train_data_list = []
        self.vector_idx_dic_list = []
        self.token_id_dic = {}
        self.pos_id_dic = {}
        self.tag_id_dic = {}

    def _set_train_data_(self, filename):
        self.train_data_list = []
        train_data = []
        f = open(filename, "r")
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                self.train_data_list.append(train_data)
                train_data = []
                continue
            elif line[0] == ';' or line[0] == '$':
                continue
            tokens = line.split("\t")
            word_idx = eval(tokens[0])
            token = tokens[1]
            pos = tokens[2]
            tag = tokens[3]
            train_data.append((word_idx, token, pos, tag))
        f.close()

    def train(self, filename):
        self._set_train_data_(filename)
        self._convert_vector_()

        print("# of train sentence : {}".format(len(self.train_data_list)))
        print("# of unique tokens : {}".format(len(self.token_id_dic)))
        print("# of unique pos : {}".format(len(self.pos_id_dic)))
        print("# of unique tag : {}".format(len(self.tag_id_dic)))

    def _convert_vector_(self):
        self._build_vector_idx_()
        self._build_seq_vector_()

    def _build_vector_idx_(self):
        self.vector_idx_dic_list = []
        for train_data in self.train_data_list:
            token_id_list = []
            pos_id_list = []
            tag_id_list = []
            for word_idx, token, pos, tag in train_data:
                token_id = _convert_source_to_idx_(token, self.token_id_dic)
                pos_id = _convert_source_to_idx_(pos, self.pos_id_dic)
                tag_id = _convert_source_to_idx_(tag, self.tag_id_dic)
                token_id_list.append(token_id)
                pos_id_list.append(pos_id)
                tag_id_list.append(tag_id)
            self.vector_idx_dic_list.append(
                {
                    "token_id_list": token_id_list,
                    "pos_id_list": pos_id_list,
                    "tag_id_list": tag_id_list
                }
            )

    def _build_seq_vector_(self):
        token_vector_size = len(self.token_id_dic)
        pos_vector_size = len(self.pos_id_dic)
        tag_vector_size = len(self.tag_id_dic)
        self.vector_seq_list = []
        for vector_idx_dic in self.vector_idx_dic_list:
            # token_vector = self._get_bow_vector(vector_idx_dic["token_id_list"], token_vector_size)
            # pos_vector = self._get_bow_vector(vector_idx_dic["pos_id_list"], pos_vector_size)
            # tag_vector = self._get_bow_vector(vector_idx_dic["tag_id_list"], tag_vector_size)
            token_vector_seq = self._get_onehot_seq_vector(vector_idx_dic["token_id_list"], token_vector_size)
            pos_vector_seq = self._get_onehot_seq_vector(vector_idx_dic["pos_id_list"], pos_vector_size)
            tag_vector_seq = self._get_onehot_seq_vector(vector_idx_dic["tag_id_list"], tag_vector_size)
            self.vector_seq_list.append((token_vector_seq, pos_vector_seq, tag_vector_seq))


    def _get_bow_vector(self, vector_idx_list, vector_size):
        vector = np.zeros(vector_size)
        for vector_idx in vector_idx_list:
            vector[vector_idx] = 1
        return vector

    def _get_onehot_seq_vector(self, vector_idx_list, vector_size):
        vector_seq = []
        for vector_idx in vector_idx_list:
            vector = np.zeros(vector_size)
            vector[vector_idx] = 1
            vector_seq.append(vector)
        return vector_seq

training_filepath = "/Users/shin285/shineware/data/hclt2016_ner/distribution2016/txt/train.txt"
trainer = Trainer()
trainer.train(training_filepath)
