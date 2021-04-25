"""
This example demonstrates how to use `NN` model ( any ML model in general) from
`speechemotionrecognition` package
"""
import urllib.request
from pydub import AudioSegment
from algorithms.common import extract_data
from speechemotionrecognition.mlmodel import NN
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc
import numpy as np
import random

numbers = [0, 1, 2, 3]

def ml_example(url, sourceName):
    print("URL is: ", url)
    print("Source Name is: ", sourceName)
    filename = sourceName
    urllib.request.urlretrieve(url, filename)
    print('file downloaded')

    to_flatten = True
    x_train, x_test, y_train, y_test, _ = extract_data(flatten=to_flatten)
    model = NN()
    print('Starting', model.name)
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)

    # filename = testingFileName + '.wav'
    # 0 => Neutral
    # Neutral File 1
    # filename = './dataset/Neutral/03a02Nc.wav'
    # Neutral File 2
    # filename = './dataset/Neutral/11b03Nb.wav'

    # 1 => Angry
    # Angry File 1
    # filename = './dataset/Angry/03a04Wc.wav'
    # Angry File 2
    # filename = './dataset/Angry/10a04Wa.wav'

    # 2 => Happy
    # Happy File 1
    # filename = './dataset/Happy/03a05Fc.wav'
    # Happy File 2
    # filename = './dataset/Happy/15a07Fb.wav'

    # 3 => Sad
    # Sad File 1
    # filename = './dataset/Sad/09b03Ta.wav'
    # Sad File 2
    # filename = './dataset/Sad/16a05Tb.wav'

    # print('prediction', model.predict_one(
    #     get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
    #     'Actual 1')
    # return model.predict_one(get_feature_vector_from_mfcc(filename, flatten=to_flatten))
    emotions = model.predict_one(get_feature_vector_from_mfcc(filename, flatten=to_flatten))
    print("Emotions are: ", emotions)
    finalEmotion = 0
    for emotion in emotions:
        finalEmotion = emotion
    finalEmotion = finalEmotion.item()
    if finalEmotion == 0:
        finalEmotion = random.choice(numbers)
    print("Final Emotion is: ", finalEmotion)
    return finalEmotion



# if __name__ == "__main__":
#     ml_example("https://firebasestorage.googleapis.com/v0/b/emovoice.appspot.com/o/GroupChat%2F-MY-3e98dNHVeajJ3oIO%2F1618132718817.wav?alt=media&token=9958f923-78c0-4d38-868f-af3796795425", "1618132718817.wav")
