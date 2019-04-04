class GlobalVar:
    train_rela_files = None
    train_ques_file = None
    train_label_file = None
    test_rela_files = None
    test_ques_file = None
    test_label_file = None
    preprocessWordVector_files = None
    preprocessWordVector_path = None
    MAX_NB_WORDS = None
    EMBEDDING_DIM = None
    LSTM_DIM = None
    NUM_FILTERS = None
    filter_sizes = None

    ques_maxlen = None
    relation_maxlen = None
    NUM_OF_RELATIONS=None

def set_train_rela_files(train_rela_files):
  GlobalVar.train_rela_files = train_rela_files
def get_train_rela_files():
  return GlobalVar.train_rela_files

def set_train_ques_file(train_ques_file):
  GlobalVar.train_ques_file = train_ques_file
def get_train_ques_file():
  return GlobalVar.train_ques_file

def set_train_label_file(train_label_file):
  GlobalVar.train_label_file = train_label_file
def get_train_label_file():
  return GlobalVar.train_label_file

def set_test_rela_files(test_rela_files):
  GlobalVar.test_rela_files = test_rela_files
def get_test_rela_files():
  return GlobalVar.test_rela_files

def set_test_ques_file(test_ques_file):
  GlobalVar.test_ques_file = test_ques_file
def get_test_ques_file():
  return GlobalVar.test_ques_file

def set_test_label_file(test_label_file):
  GlobalVar.test_label_file = test_label_file
def get_test_label_file():
  return GlobalVar.test_label_file

def set_preprocessWordVector_files(preprocessWordVector_files):
  GlobalVar.preprocessWordVector_files = preprocessWordVector_files
def get_preprocessWordVector_files():
  return GlobalVar.preprocessWordVector_files

def set_preprocessWordVector_path(preprocessWordVector_path):
  GlobalVar.preprocessWordVector_path = preprocessWordVector_path
def get_preprocessWordVector_path():
  return GlobalVar.preprocessWordVector_path

def set_MAX_NB_WORDS(MAX_NB_WORDS):
  GlobalVar.MAX_NB_WORDS = MAX_NB_WORDS
def get_MAX_NB_WORDS():
  return GlobalVar.MAX_NB_WORDS

def set_EMBEDDING_DIM(EMBEDDING_DIM):
  GlobalVar.EMBEDDING_DIM = EMBEDDING_DIM
def get_EMBEDDING_DIM():
  return GlobalVar.EMBEDDING_DIM

def set_LSTM_DIM(LSTM_DIM):
  GlobalVar.LSTM_DIM = LSTM_DIM
def get_LSTM_DIM():
  return GlobalVar.LSTM_DIM

def set_NUM_FILTERS(NUM_FILTERS):
  GlobalVar.NUM_FILTERS = NUM_FILTERS
def get_NUM_FILTERS():
  return GlobalVar.NUM_FILTERS

def set_filter_sizes(filter_sizes):
  GlobalVar.filter_sizes = filter_sizes
def get_filter_sizes():
  return GlobalVar.filter_sizes

def set_ques_maxlen(ques_maxlen):
  GlobalVar.ques_maxlen = ques_maxlen
def get_ques_maxlen():
  return GlobalVar.ques_maxlen

def set_relation_maxlen(relation_maxlen):
  GlobalVar.relation_maxlen = relation_maxlen
def get_relation_maxlen():
  return GlobalVar.relation_maxlen

def set_NUM_OF_RELATIONS(NUM_OF_RELATIONS):
  GlobalVar.NUM_OF_RELATIONS = NUM_OF_RELATIONS
def get_NUM_OF_RELATIONS():
  return GlobalVar.NUM_OF_RELATIONS