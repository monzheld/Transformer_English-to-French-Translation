import os
import sys
import tensorflow as tf

translator = tf.saved_model.load(os.path.join(os.getcwd(), "saved_model/translator"))

sentence = str(sys.stdin.readline())
translator(sentence).numpy().decode("utf-8")
