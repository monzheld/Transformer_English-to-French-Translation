import os
import logging
import tensorflow as tf

from data_loader import Tokenization

logging.getLogger('tensorflow').setLevel(logging.ERROR)  

tokenizer = Tokenization()
tokenizer_src, tokenizer_trg = tokenizer.load_tokenizer()

transformer = tf.saved_model.load(os.path.join(os.getcwd(), "saved_model/best_model"))


class Translator(tf.Module):
    def __init__(self, tokenizer_src, tokenizer_trg, transformer):
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trg = tokenizer_trg
        self.transformer = transformer
    
    def __call__(self, sentence, max_length=100): 
        # Add start and end token to input sentence
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        
        sentence = self.tokenizer_src.tokenize(sentence).to_tensor() 

        encoder_input = sentence

        # the first token given to the transformer model should be the start token of target 
        start_end = self.tokenizer_trg.tokenize([''])[0] 
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)

            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break
        
        output = tf.transpose(output_array.stack()) # (1, tokens)

        text = self.tokenizer_trg.decode(output)[0] 

        tokens = self.tokenizer_trg.encode(output)[0] 

        _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

        return text, tokens, attention_weights


translator = Translator(tokenizer_src, 
                        tokenizer_trg, 
                        transformer)


# Export translator 
class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        (result,
         tokens,
         attention_weights) = self.translator(sentence, max_length=100)
        
        return result


ex_translator = ExportTranslator(translator)

# Save the exported translator 
tf.saved_model.save(ex_translator, export_dir=os.path.join(os.getcwd(), "saved_model/translator"))
