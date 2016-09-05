from nltk import FreqDist # used later to plot and get count
from nltk.tokenize import word_tokenize # tokenizes our sentence by word
from tkinter.filedialog import askopenfilename
import os


#fdist = FreqDist(word.lower() for word in word_tokenize(text))

#tknz = word_tokenize(text)
#fdist = FreqDist(tknz)






datapath = askopenfilename(initialdir= os.getcwd(), title="Select ASR file")
file = open(datapath)
lines = [line.strip().split('\t') for line in file]

text = ""

for line in lines:
    print(line)
    text += line[1]
    text += line[2]


print(text)

#fdist = FreqDist(word.lower() for word in word_tokenize(text))
#fdist.plot()