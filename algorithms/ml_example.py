"""
This example demonstrates how to use `NN` model ( any ML model in general) from
`speechemotionrecognition` package
"""
import urllib.request
from pydub import AudioSegment
from common import extract_data
from speechemotionrecognition.mlmodel import NN
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


def ml_example():
    fileUrl = 'https://firebasestorage.googleapis.com/v0/b/emovoice.appspot.com/o/ChatAudios%2F169166106166136108205128219127%2F1616835089442.m4a?alt=media&token=5ff44ead-fd91-4186-ad53-8c18df92edb6'
    name = '1613979505836'
    filename = name + '.m4a'
    urllib.request.urlretrieve(fileUrl, filename)
    print('file downloaded')
    wma_version = AudioSegment.from_file(filename, 'm4a')
    wma_version.export(name + '.wav', format="wav")
    print('file converted')

    to_flatten = True
    x_train, x_test, y_train, y_test, _ = extract_data(flatten=to_flatten)
    model = NN()
    print('Starting', model.name)
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)

    # 0 => Neutral
    # Neutral File 1
    # filename = './dataset/Neutral/03a02Nc.wav'
    # Neutral File 2
    # filename = './dataset/Neutral/11b03Nb.wav'

    # 1 => Angry
    # Angry File 1
    filename = './dataset/Angry/03a04Wc.wav'
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

    print('prediction', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
        'Actual 1')


if __name__ == "__main__":
    ml_example()
