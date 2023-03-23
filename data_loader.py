import os
import urllib3
import zipfile
import shutil
import re
import unicodedata
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.utils import pad_sequences

class TranslationDataset():
  def __init__(self):
    self.data = self.get_dataset()
    self.train, self.val, self.test = self.split_dataset(self.data)

    print(f'Dataset size: {len(self.data)}')
    print(f'Train set size: {len(self.train)}')
    print(f'Validation set size: {len(self.val)}')
    print(f'Test set size: {len(self.test)}', '\n')

  def get_dataset(self):
    http = urllib3.PoolManager()
    url ='http://www.manythings.org/anki/fra-eng.zip'
    filename = 'fra-eng.zip'
    path = os.getcwd()
    zipfilename = os.path.join(path, filename)
    user_agent = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}

    with http.request('GET', url, preload_content=False, headers=user_agent) as r, open(zipfilename, 'wb') as out_file:   
      shutil.copyfileobj(r, out_file)
    
    with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
      zip_ref.extractall()
    
    fra_eng_csv = pd.read_csv('fra.txt', names=['SRC', 'TRG', 'lic'], sep='\t')
    del fra_eng_csv['lic']
    data = fra_eng_csv.loc[:, 'SRC':'TRG']

    os.remove(zipfilename)
    os.remove(os.path.join(path, '_about.txt'))
    os.remove(os.path.join(path, 'fra.txt'))

    return data
  
  def split_dataset(self, data=None):
    if data is None:
      data = self.get_dataset()
    # Shuffle data
    data = data.sample(frac=1, random_state=42)

    # Split into train/val/test (80:10:10)
    train, test = train_test_split(data, test_size=0.1, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    return train, val, test 


class Preprocess():
  def __init__(self):
    super(Preprocess, self).__init__()
    dataset = TranslationDataset()
    self.train, self.val, self.test = dataset.split_dataset()

  def to_ascii(self, s):
    # Remove accents in french
    # ex) 'déjà diné' -> deja dine
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                      if unicodedata.category(c) != 'Mn')
  
  def preprocess_sentence(self, df):
    for col in df.columns:
      # Remove accents and convert to lower case
      df[col] = df[col].apply(lambda x: self.to_ascii(x.lower()))
      # Add spaces between words and punctuation
      # ex) "I am a student." => "I am a student ."
      df[col] = df[col].apply(lambda x: re.sub(r"([?.!,¿])", r" \1", x))
      # Convert to spaces except (a-z, A-Z, ".", "?", "!", ",")
      df[col] = df[col].apply(lambda x: re.sub(r"[^a-zA-Z!.?]+", r" ", x))
      # Convert multiple spaces to single space
      df[col] = df[col].apply(lambda x: re.sub(r"\s+", " ", x))
    return df
  
  def preprocess(self):
    preprocessed_train = self.preprocess_sentence(self.train)
    preprocessed_val = self.preprocess_sentence(self.val)
    preprocessed_test = self.preprocess_sentence(self.test)

    # Convert to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((preprocessed_train['SRC'], preprocessed_train['TRG']))
    val_dataset = tf.data.Dataset.from_tensor_slices((preprocessed_val['SRC'], preprocessed_val['TRG']))
    test_dataset = tf.data.Dataset.from_tensor_slices((preprocessed_test['SRC'], preprocessed_test['TRG']))

    return train_dataset, val_dataset, test_dataset


class Tokenization():
  def __init__(self):
    super(Tokenization, self).__init__()
    preprocessed = Preprocess()
    self.train_dataset, self.val_dataset, self.test_dataset = preprocessed.preprocess()

    self.tokenizer_src, self.tokenizer_trg = self.create_tokenizer(self.train_dataset)
  
  def create_tokenizer(self, train_dataset):
    # Create tokenizers with train_dataset
    tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (src.numpy() for src, _ in train_dataset), target_vocab_size=2**13)
    
    tokenizer_trg = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (trg.numpy() for _, trg in train_dataset), target_vocab_size=2**13)
    
    return tokenizer_src, tokenizer_trg
  
  def save_tokenizer(self):
    save_path = os.path.join(os.getcwd(), "tokenizer")
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    
    # Save tokenizers
    self.tokenizer_src.save_to_file(os.path.join(save_path, "tokenizer_src"))
    self.tokenizer_trg.save_to_file(os.path.join(save_path, "tokenizer_trg"))

  def load_tokenizer(self):
    save_path = os.path.join(os.getcwd(), "tokenizer")
    if os.path.exists(save_path):
      # Load tokenizers
      self.tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.load_from_file(os.path.join(save_path, "tokenizer_src"))
      self.tokenizer_trg = tfds.deprecated.text.SubwordTextEncoder.load_from_file(os.path.join(save_path, "tokenizer_trg"))

    return self.tokenizer_src, self.tokenizer_trg
  
  def encode(self, lang1, lang2):
    # Add <sos>, <eos> token
    lang1 = [self.tokenizer_src.vocab_size] + self.tokenizer_src.encode(
        lang1.numpy()) + [self.tokenizer_src.vocab_size + 1]
    
    lang2 = [self.tokenizer_trg.vocab_size] + self.tokenizer_trg.encode(
        lang2.numpy()) + [self.tokenizer_trg.vocab_size + 1]
    
    return lang1, lang2
  
  def tf_encode(self, src, trg):
    result_src, result_trg = tf.py_function(self.encode, [src, trg], [tf.int64, tf.int64])

    result_src.set_shape([None])
    result_trg.set_shape([None])

    return result_src, result_trg
  
  def add_padded_batch(self, dataset, batch_size, padded_shapes):
    return dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
  
  def add_pad_sequences(self, dataset, maxlen=100):
    return dataset.map(lambda x, y: (pad_sequences(x, maxlen=maxlen, padding='post'), 
                                   pad_sequences(y, maxlen=maxlen, padding='post')))

  def tokenization(self, batch_size, maxlen):
    # Tokenize datasets
    train_dataset = self.train_dataset.map(self.tf_encode)
    val_dataset = self.val_dataset.map(self.tf_encode)
    test_dataset = self.test_dataset.map(self.tf_encode)

    # Make batches
    train_dataset = self.add_padded_batch(train_dataset, batch_size, padded_shapes=([-1], [-1]))
    val_dataset = self.add_padded_batch(val_dataset, batch_size, padded_shapes=([-1], [-1]))
    test_dataset = self.add_padded_batch(test_dataset, batch_size, padded_shapes=([-1], [-1]))

    # Pad sequences
    train_dataset = self.add_pad_sequences(train_dataset, maxlen)
    val_dataset = self.add_pad_sequences(val_dataset, maxlen)
    test_dataset = self.add_pad_sequences(test_dataset, maxlen)

    return train_dataset, val_dataset, test_dataset
  
  def vocab_size(self):
    input_vocab_size = self.tokenizer_src.vocab_size + 2
    target_vocab_size = self.tokenizer_trg.vocab_size + 2
    return input_vocab_size, target_vocab_size