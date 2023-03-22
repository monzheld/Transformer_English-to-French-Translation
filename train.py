import os
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import BayesianOptimization

from model.transformer import Transformer
from data_loader import Tokenization

tokenizer = Tokenization()
input_vocab_size, target_vocab_size = tokenizer.vocab_size()

train_dataset, val_dataset, _ = tokenizer.tokenization(batch_size=64, maxlen=100)

EPOCHS = 100 


class BLEUCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(BLEUCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        # Get predictions on validation data
        predictions = self.model.predict(self.validation_data[0])

        # Convert predictions and targets to text
        targets = []
        for t in self.validation_data[1]:
            text = " ".join([str(i) for i in t if i != 0])
            targets.append(text)

        predictions = self.convert_predictions(predictions)

        # Calculate BLEU score
        score = self.bleu_score(targets, predictions)
        logs['val_bleu_score'] = score

    def convert_predictions(self, predictions):
        """Converts predictions to text."""
        results = []
        for prediction in predictions:
            text = " ".join([str(i) for i in np.argmax(prediction, axis=1) if i != 0])
            results.append(text)
        return results

    def bleu_score(self, targets, predictions):
        """Calculates BLEU score."""
        total_score = 0.0
        for target, prediction in zip(targets, predictions):
            total_score += self.sentence_bleu([target.split()], prediction.split())
        return total_score / len(targets)

    def sentence_bleu(self, references, hypothesis):
        """Calculates sentence-level BLEU score."""
        max_n = len(hypothesis)
        candidate = hypothesis[:]
        references = [[ref[:i + 1] for i in range(len(ref))] for ref in references]
        precisions = np.zeros(max_n)
        for n in range(max_n):
            cand_ngrams = self._get_ngrams(candidate, n + 1)
            cand_count = len(cand_ngrams)
            ref_count = max([len([1 for ref_ngrams in ref if ref_ngrams == cand_ngrams])
                             for ref in references])
            precisions[n] = ref_count / cand_count if cand_count > 0 else 0.0
        geo_mean = np.exp(np.log(precisions[precisions > 0]).mean())
        brevity_penalty = self._brevity_penalty(candidate, references)
        return geo_mean * brevity_penalty

    def _get_ngrams(self, text, n):
        """Returns a list of n-grams for the given text."""
        return [tuple(text[i:i + n]) for i in range(len(text) - n + 1)]

    def _brevity_penalty(self, candidate, references):
        """Calculates brevity penalty."""
        c = len(candidate)
        r = min(abs(len(candidate) - len(ref)) for ref in references)
        if c > r:
            return 1.0
        else:
            return np.exp(1 - r / c)


def build_model(hp):
    # Set hyperparameters
    num_layers = hp.Int('num_layers', min_value=1, max_value=8, step=1)
    d_model = hp.Int('d_model', min_value=32, max_value=512, step=32)
    num_heads = hp.Int('num_heads', min_value=2, max_value=10, step=2)
    dff = hp.Int('dff', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)

    # Define the learning rate schedule
    def learning_rate_schedule(d_model, warmup_steps, initial_lr):
        """Applies a linear warmup and a fixed decay to the learning rate."""
        def lr_function(step):
            step = tf.cast(step + 1, dtype=tf.float32)
            lr = initial_lr * tf.math.minimum(1.0, step / warmup_steps)
            lr /= tf.math.sqrt(tf.cast(d_model, dtype=tf.float32))
            return lr
        return lr_function

    # Define the Transformer model
    model = Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        input_vocab_size=input_vocab_size, 
                        target_vocab_size=target_vocab_size, 
                        pe_input=1000, 
                        pe_target=1000, 
                        rate=dropout_rate)

    # Define the optimizer
    initial_lr = hp.Choice('initial_lr', values=[1e-2, 1e-3, 5e-4, 1e-4])
    warmup_steps = hp.Int('warmup_steps', min_value=4000, max_value=20000, step=2000)
    lr_fn = learning_rate_schedule(d_model, warmup_steps, initial_lr)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy')

    return model


# Define the tuner
tuner = BayesianOptimization(build_model,
                             objective='val_bleu_score',
                             max_trials=20, 
                             directory=os.getcwd(),
                             project_name='transformer_bo')

# Start the hyperparameter search
bleu_callback = BLEUCallback(val_dataset)

callbacks = [
    bleu_callback,
    tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, 
                                      logs: tqdm.write(f"Epoch: {epoch+1}, Loss: {logs['loss']:.4f}, BLEU Score: {logs['val_bleu_score']:.4f}"))
]

tuner.search(x=train_dataset,
             epochs=EPOCHS,
             validation_data=val_dataset,
             callbacks=callbacks)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the best model
best_model = build_model(best_hps)

# Save the best model
tf.saved_model.save(best_model, export_dir=os.path.join(os.getcwd(), "saved_model/best_model"))
