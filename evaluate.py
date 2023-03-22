import os
from tqdm import tqdm
import tensorflow as tf

from data_loader import Tokenization

tokenizer = Tokenization()
_, _, test_dataset = tokenizer.tokenization(batch_size=64, maxlen=100)

model = tf.saved_model.load(os.path.join(os.getcwd(), "saved_model/best_model"))

# Evaluate test dataset 
test_loss, test_bleu_score = model.evaluate(test_dataset, verbose=1)
print(f"Test loss: {test_loss:.4f}, Test BLEU score: {test_bleu_score:.4f}")
