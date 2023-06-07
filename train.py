import os
import logging
from tqdm import tqdm
import numpy as np

import tensorflow as tf
from tensorflow import keras

from model.transformer import Transformer
from data_loader import Tokenization

logging.getLogger('tensorflow').setLevel(logging.ERROR) 

tokenizer = Tokenization()
input_vocab_size, target_vocab_size = tokenizer.vocab_size()

train_dataset, val_dataset, _ = tokenizer.tokenization(buffer_size=20000, batch_size=64)

EPOCHS = 30

# Set hyperparameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

# Define the learning rate schedule
def learning_rate_schedule(d_model, warmup_steps, initial_lr):
    """Applies a linear warmup and a fixed decay to the learning rate."""
    def lr_function(step):
        step = tf.cast(step + 1, dtype=tf.float32)
        lr = initial_lr * tf.math.minimum(1.0, step / warmup_steps)
        lr /= tf.math.sqrt(tf.cast(d_model, dtype=tf.float32))
        return lr
    return lr_function

initial_lr = 5e-4
warmup_steps = 10000
lr_fn = learning_rate_schedule(d_model, warmup_steps, initial_lr)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn(0), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Define loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

# Define train loss metric
train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
bleu_score = tf.keras.metrics.Mean(name='bleu_score')

# Define train step function
input_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=input_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = model([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

# Define validation step function
import nltk
from nltk.translate.bleu_score import sentence_bleu

@tf.function(input_signature=input_signature)
def val_step(inp, tar, bleu_score):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    predictions, _ = model(inp, tar_inp, False)
    loss = loss_function(tar_real, predictions)
    val_loss(loss)

    # Convert predictions and targets to text
    targets = []
    for t in tar.numpy():
        text = " ".join([str(i) for i in t if i != 0])
        targets.append(text)

    predictions = []
    for p in predictions.numpy():
        text = " ".join([str(i) for i in np.argmax(p, axis=1) if i != 0])
        predictions.append(text)

    # Calculate BLEU score
    for target, prediction in zip(targets, predictions):
        score = sentence_bleu([target.split()], prediction.split())
        bleu_score.update_state(score)

    return bleu_score

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

best_val_bleu = 0.0  # Track the best validation BLEU score seen so far
best_model = None  # Store the best model

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    val_loss.reset_states()
    bleu_score.reset_states()

    # Train loop
    for (batch, (inp, targ)) in enumerate(train_dataset):
        train_step(inp, targ)

    # Validation loop
    for (batch, (inp, targ)) in enumerate(val_dataset):
        bleu_score = val_step(inp, targ, bleu_score)

    # Print epoch results
    tqdm.write(f"Epoch: {epoch+1}, Train Loss: {train_loss.result():.4f}, Val Loss: {val_loss.result():.4f}, Val BLEU Score: {bleu_score.result():.4f}")

    # Save the best model
    if bleu_score.result() > best_val_bleu:
        best_val_bleu = bleu_score.result()
        best_model = model

# Save the best model
tf.saved_model.save(best_model, export_dir=os.path.join(os.getcwd(), "saved_model/best_model"))
