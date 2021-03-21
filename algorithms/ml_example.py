"""
This example demonstrates how to use `NN` model ( any ML model in general) from
`speechemotionrecognition` package
"""
from common import extract_data
from speechemotionrecognition.mlmodel import NN
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


def ml_example():
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
