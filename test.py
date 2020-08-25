import argparse
import numpy as np
import pickle
import time
import sys

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--input", type=str, default='input.txt',
    help="path to input file")
ap.add_argument("-m", "--model", type=str, default='sentiment.model',
    help="path to output model")
ap.add_argument("-d", "--dataset", type=str, default='dataset.csv',
    help="path to input dataset")
ap.add_argument("-l", "--live", type=bool, default=False,
    help="whether to run live")
args = vars(ap.parse_args())

maxlen=200

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args['model'])
print("[INFO] network loaded!")

tokenizer = pickle.load(open( "tokenizer.p", "rb" ))

def predict(line):
    #start = time.perf_counter()
    stripped_input = line.strip() #remove newline
    instance = tokenizer.texts_to_sequences(stripped_input)
    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    result = ""
    sentiment = model.predict(instance)
    #end = time.perf_counter()
    #timer_in_ms = float(round((end-start)*1000))
    result = "✓ positive" if sentiment[0][0] > 0.5 else "x negative"
    print("{} | '{}'".format(result, stripped_input))


if args["live"]==True:
    print("[INFO] live mode, type below..\n")
    while True:
        line = sys.stdin.readline()
        print("\n=====RESULT=====")
        predict(line)

else: 
    with open(args["input"], "r") as input_file:
      for line in input_file:
        predict(line)
