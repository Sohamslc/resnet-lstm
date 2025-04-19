import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import cv2
from glob import glob

# Load models and vocab
resnet50 = tf.keras.models.load_model('resnet50.h5')
lstm = tf.keras.models.load_model('lstm.h5')
MAX_LEN = 34
count_words = np.load('vocab.npy', allow_pickle=True).item()
inverse_dict = {val: key for key, val in count_words.items()}

def preprocess_image(test_img_path):
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, (1, 224, 224, 3))
    return img

def predict_caption(image_path):
    # image = preprocess_image(image_path)
    image = image_path
    features = resnet50.predict(image).reshape(1, 2048)
    pred_text = ['startofseq']
    caption = ''
    
    for _ in range(25):
        encoded = [count_words.get(word, 0) for word in pred_text]
        encoded = pad_sequences([encoded], maxlen=MAX_LEN, padding='post', truncating='post')
        pred_idx = np.argmax(lstm.predict([features, encoded]))
        sampled_word = inverse_dict[pred_idx]
        if sampled_word == 'endofseq':
            break
        caption += ' ' + sampled_word
        pred_text.append(sampled_word)
    
    return caption.strip()
