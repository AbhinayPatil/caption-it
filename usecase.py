import os
import cv2
import numpy as np
import tensorflow as tf

#use this code with already trained model(best_model.h5)
from PIL import Image
import matplotlib.pyplot as plt

def generate_caption(image_name):
    model = tf.keras.models.load_model("best_model.h5")
    
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
        
    all_captions[:10]
    # tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1


    # get maximum length of the caption available
    max_length = max(len(caption.split()) for caption in all_captions)
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    # image_id = image_name.split('.')[0]
    #img_path = os.path.join('./Flickr8k_Dataset',image_id+'.jpg')        
    img_path = os.path('./image.jpg')
    image = Image.open(img_path)
    # captions = mapping[image_id]
    # print('---------------------Actual---------------------')
    # for caption in captions:
    #     print(caption)
    # predict the caption
    y_pred = predict_caption(model, image, tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text
    
generate_caption("image.jpg")