import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    # masking <PAD> tokens in source sentences 
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    # masking future tokens 
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights

    Args:
        |q| = (..., seq_len_q, depth) 
        |k| = (..., seq_len_k, depth) 
        |v| = (..., seq_len_v, depth_v) 
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults: None

    Returns:
        |output| = (..., seq_len_q, depth_v)
        |attention_weights| = (..., seq_len_q, seq_len_k)
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # dk: dimension of key vectors (= d_model / num_heads)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # masking
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)  # depth_v = dv = dk = d_model / num_heads

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads 

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):

        # Args are v, k, q, not q, k, v

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled dot product attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # |scaled_attention| = (batch_size, num_heads, seq_len_q, depth)
        # |attention_weights| = (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        # concatenate heads
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])


class PreLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(PreLayerNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        d_model = input_shape[-1]
        self.gamma = self.add_weight("gamma", shape=[d_model], initializer="ones")
        self.beta = self.add_weight("beta", shape=[d_model], initializer="zeros")
        super(PreLayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean = tf.keras.backend.mean(inputs, axis=-1, keepdims=True)
        variance = tf.keras.backend.mean(tf.keras.backend.square(inputs - mean), axis=-1, keepdims=True)
        norm_inputs = (inputs - mean) / tf.keras.backend.sqrt(variance + self.epsilon)
        output = self.gamma * norm_inputs + self.beta
        return output


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderBlock, self).__init__()

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = PreLayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate)

        # Position-wise FFNN
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm2 = PreLayerNormalization()
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):

        # Post-LN: 

        # attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        # attn_output = self.dropout1(attn_output, training=training)
        ## self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        # ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        # ffn_output = self.dropout2(ffn_output, training=training)
        ## self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)


        # Pre-LN:

        # Layer Normalization
        attn_output = self.layernorm1(x) 
        # Multi-Head Attention
        attn_output, _ = self.mha(attn_output, attn_output, attn_output, mask)  # (batch_size, input_seq_len, d_model)
        # Dropout
        attn_output = self.dropout1(attn_output, training=training)
        # Residual connection
        out1 = x + attn_output

        # Layer Normalization
        ffn_output = self.layernorm2(out1)
        # Position-wise FFNN
        ffn_output = self.ffn(ffn_output)  
        # Dropout
        ffn_output = self.dropout2(ffn_output, training=training)
        # Residual connection
        out2 = out1 + ffn_output

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderBlock, self).__init__()

        # self-attention in Decoder
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = PreLayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate)

        # Encoder-Decoder attention
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.layernorm2 = PreLayerNormalization()
        self.dropout2 = tf.keras.layers.Dropout(rate)

        # Position-wise FFNN
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm3 = PreLayerNormalization()
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # |enc_output| = (batch_size, input_seq_len, d_model)

        # Post-LN: 

        # attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        # attn1 = self.dropout1(attn1, training=training)
        ## self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # out1 = self.layernorm1(attn1 + x)

        # attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        # attn2 = self.dropout2(attn2, training=training)
        ## self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        # ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        # ffn_output = self.dropout3(ffn_output, training=training)
        ## self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)


        # Pre-LN:

        # Layer Normalization
        attn1 = self.layernorm1(x)
        # Multi-Head Attention -> mask: look_ahead_mask
        attn1, attn_weights_block1 = self.mha1(attn1, attn1, attn1, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        # Dropout
        attn1 = self.dropout1(attn1, training=training)
        # Residual connection
        out1 = x + attn1

        # Layer Normalization
        normed_v_k = self.layernorm2(enc_output) # enc_output = v, k
        normed_q = self.layernorm2(out1) # out1 = q 
        # Multi-Head Attention -> mask: padding_mask
        attn2, attn_weights_block2 = self.mha2(normed_v_k, normed_v_k, normed_q, padding_mask)  # (batch_size, target_seq_len, d_model)
        # Dropout
        attn2 = self.dropout2(attn2, training=training)
        # Residual connection
        out2 = out1 + attn2

        # Layer Normalization
        ffn_output = self.layernorm3(out2)
        # Position-wise FFNN
        ffn_output = self.ffn(ffn_output)  # (batch_size, target_seq_len, d_model)
        # Dropout
        ffn_output = self.dropout3(ffn_output, training=training)
        # Residual connection
        out3 = out2 + ffn_output

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderBlock(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # Embedding layer 
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # add positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        # Dropout
        x = self.dropout(x, training=training)

        # Stack Encoder Blocks
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderBlock(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Embedding layer 
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # add positional encoding
        x += self.pos_encoding[:, :seq_len, :]
        # Dropout
        x = self.dropout(x, training=training)

        # Stack Decoder Blocks
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        # |x| = (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs, training):
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        # |dec_output| = (batch_size, tar_seq_len, d_model)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
    
    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Decoder padding mask
        dec_padding_mask = create_padding_mask(inp)

        # look ahead mask
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar) # including padding mask
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask
