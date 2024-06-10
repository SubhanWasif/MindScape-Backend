import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras
from tensorflow.keras import layers
from tensorflow.keras import layers
import keras.ops as ops



from flask import Flask ,request
from flask_cors import CORS




import pickle

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('val_concept_embeddings.pkl', 'rb') as handle:
    val_concept_embeddings = pickle.load(handle)
with open('glove_embeddings.pkl', 'rb') as handle:
    glove_embeddings = pickle.load(handle)
with open('val_df.pkl', 'rb') as handle:
    val_df = pickle.load(handle)

# Now you can use the 'data' object as per your requirements
    

eeg_val = np.array(val_df['eeg_signal'].tolist()) #(17, 100)
val_captions = val_df['image_captions'].tolist() 
val_concepts = val_df['image_concepts'].tolist()




vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100  # Dimensionality of GloVe embeddings

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector 
        
        
        


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="gelu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            #print("I am here")
            #attention_mask = ops.cast(mask[:, None, :], dtype="int32")
            #attention_mask = mask[:, :, tf.newaxis] #expands masks to (B, 56, 1)
            #print(attention_mask.shape)
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
        else:
            #print("I Shoudn't be here")
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix), trainable=False
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, concept_embed, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = ops.cast(mask[:, None, :], dtype="int32")
            padding_mask = ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        
        #Reshape and Concatenate Concept Embeddings
        concept_embed = keras.layers.RepeatVector(18)(concept_embed)
        out_3 = keras.layers.Concatenate()([out_2, concept_embed])
        
        proj_output = self.dense_proj(out_3)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = ops.arange(sequence_length)[:, None]
        j = ops.arange(sequence_length)
        mask = ops.cast(i >= j, dtype="int32")
        mask = ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = ops.concatenate(
            [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])],
            axis=0,
        )
        return ops.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


custom_objects= {
    'TransformerEncoder': TransformerEncoder,
    'PositionalEmbedding': PositionalEmbedding,
    'TransformerDecoder': TransformerDecoder,
}

model = load_model('transformer.keras', custom_objects=custom_objects)



app = Flask(__name__)
CORS(app)


@app.route('/', methods=["GET"])
def home():
    return 'Hello POST World'


@app.route('/predict', methods=["POST"])
def prediction():
    # For POST request, expecting JSON data in the body
    if (request.method == 'POST'):
        prompt = request.get_json()
        data = prompt['prompt']
        print(data)
        index = val_captions.index(data)
        print(index)
        eeg_sample=eeg_val[index]
        concept_sample = val_concept_embeddings[index]
        start_token = tokenizer.word_index['<start>']
        sequence = [start_token]
        for _ in range(18 - 1):
            padded_sequence = pad_sequences([sequence], maxlen=18, padding='post')
            predictions = model.predict([eeg_sample.reshape(1, 17, 100), padded_sequence, concept_sample.reshape(1, 100)])
            next_index = np.argmax(predictions[0, len(sequence)-1, :])
            if next_index == tokenizer.word_index['<end>']:
                break
            sequence.append(next_index)
        generated_sequence = ' '.join(tokenizer.index_word[i] for i in sequence[1:] if i > 0)
        return {'generated_sequence': generated_sequence}
    else:
        return 'Invalid request method'
        
if __name__ == '__main__':

    app.run( debug = True,port=3000, host='0.0.0.0')
