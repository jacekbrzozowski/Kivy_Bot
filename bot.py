import nltk
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os
import time


with open("intents.json") as file:
    data = json.load(file)

try:
    with open ("data.pickle", "rb") as f: #rb-read bytes
        words, labels, training, output = pickle.load(f) #sejwujemy te listy poniewaz
#pozniej je wykorzystujemy i nie chcemy zeby caly program sie czytal od poczatku do konca
        
except:
    words = []
    labels = []
    docs_pattern = []
    docs_tag = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_pattern.append(wrds)
            docs_tag.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

            
    """Sprawdzamy ilosc slowek w words dlatego dajemy wszystko z malych liter, zeby
    system nie myslal ze slowa z malych i z duzych liter to inne slowa
    i nie podliczal za duzo"""
    words = [st.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words))) #usuwamy duplikaty, od razu tworzymy liste, a set usuwa duplikaty

    labels = sorted(labels)

    training = [] # lista 0 i 1 ktore okreslaja czy znajduje sie dane slowo w frazie
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for pat, doc in enumerate(docs_pattern):
        bag = []
        wrds = [st.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_tag[pat])] = 1

    #przechodzimy przez liste labels z tagami i sprawdzamy gdzie znajduje sie
    #interesujacy nas tag i wtedy ustawiamy te wartosc na 1 w naszym output_row

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training) #zamieniamy na numpy bo wymaga tego TF learns
    output = numpy.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)

tensorflow.compat.v1.reset_default_graph() #pozbywamy sie starych ustawien

net = tflearn.input_data(shape=[None, len(training[0])])#definiuje input ktorego sie spodziewamy
net = tflearn.fully_connected(net, 8)#fully connected layer to nasz neural network net
net = tflearn.fully_connected(net, 8)# 8 neuronow dla tej warstwy
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)#DNN-typ sieci neuronowej ktory wezmie wszytsko(nety) i je zmieli

if os.path.exists("model.tflearn.meta"):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [st.stem(word.lower())for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se: # aktualne slowo w words list jest takie jak w sentencji
                bag[i] = 1
    return numpy.array(bag)

def chat(self, inp): #wzywamy cha na koncu jesli chcemy wejsc w interakcje
    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = numpy.argmax(results) #wynik z najwyzszym prawdopodobienstwem
    tag = labels[results_index]#da nam label ktory pasuje do wiadomosci

    if results[results_index] > 0.7: #jesli prawdopodobienstwo danej odpowiedzi jest wieksze niz 70% to robimy kod
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return "Bot: " + random.choice(responses) + "\n"


    else:
        return "Bot: I didn`t get that, try again." + "\n"



            
    
