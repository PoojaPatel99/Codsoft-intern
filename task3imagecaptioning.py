import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Add
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
from PIL import Image
import os

# Load and preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # ResNet50 expects 224x224 input
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)  # Preprocessing for ResNet50
    return image

# Extract image features using ResNet50
def extract_image_features(image_path):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # Remove top layer for feature extraction
    image = preprocess_image(image_path)
    features = model.predict(image)
    return features

# Captioning model (Encoder-Decoder with LSTM)
def create_captioning_model(vocab_size, max_sequence_length, embedding_dim=256, lstm_units=512):
    # Encoder: Image feature extractor
    image_input = tf.keras.Input(shape=(2048,))  # ResNet50 output shape
    image_embedding = Dense(embedding_dim, activation='relu')(image_input)

    # Decoder: LSTM for generating captions
    caption_input = tf.keras.Input(shape=(max_sequence_length,))
    caption_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(caption_input)
    caption_lstm = LSTM(lstm_units)(caption_embedding)
    
    # Combine image and caption inputs
    combined = Add()([image_embedding, caption_lstm])
    output = Dense(vocab_size, activation='softmax')(combined)

    # Create and compile model
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Generate a caption for an image
def generate_caption(model, image_path, tokenizer, max_sequence_length):
    image_features = extract_image_features(image_path)
    image_features = np.reshape(image_features, (1, 2048))  # Reshape for model input
    
    # Start caption with the 'start' token
    caption_input = np.zeros((1, max_sequence_length))  # Start with a zeroed input sequence
    caption_input[0][0] = tokenizer.word_index['startseq']
    
    # Predict the next word in the sequence (loop to generate full caption)
    for i in range(1, max_sequence_length):
        predicted_word_probs = model.predict([image_features, caption_input])
        predicted_word_idx = np.argmax(predicted_word_probs)
        
        # If we predict the 'end' token, break the loop
        word = tokenizer.index_word.get(predicted_word_idx, '')
        if word == 'endseq':
            break
        
        # Add predicted word to the caption sequence
        caption_input[0][i] = predicted_word_idx
    
    # Convert sequence of word indices to words
    caption = ' '.join([tokenizer.index_word[idx] for idx in caption_input[0] if idx > 0])
    return caption

# Example of training data and how to prepare it:
def load_dataset(image_folder, caption_file):
    image_paths = []
    captions = []
    
    with open(caption_file, 'r') as file:
        for line in file:
            image_id, caption = line.split('\t')
            image_paths.append(os.path.join(image_folder, image_id + '.jpg'))
            captions.append(caption.strip())

    return image_paths, captions

# Example of pre-processing captions and training the model
def preprocess_captions(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding
    max_sequence_length = max(len(c.split()) for c in captions)
    
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    
    return tokenizer, padded_sequences, vocab_size, max_sequence_length

# Main execution: Train and use model
def main():
    image_folder = 'path/to/images'
    caption_file = 'path/to/captions.txt'
    
    # Load dataset
    image_paths, captions = load_dataset(image_folder, caption_file)
    
    # Preprocess captions
    tokenizer, padded_sequences, vocab_size, max_sequence_length = preprocess_captions(captions)
    
    # Create the model
    model = create_captioning_model(vocab_size, max_sequence_length)
    
    # Example: Train the model (not implemented fully here due to dataset size)
    # model.fit([image_features, padded_sequences], labels, epochs=10)
    
    # Example: Generate caption for a new image
    image_path = image_paths[0]  # Example image
    caption = generate_caption(model, image_path, tokenizer, max_sequence_length)
    print("Generated Caption: ", caption)

if __name__ == '__main__':
    main()
