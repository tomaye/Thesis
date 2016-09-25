import os, re, sys
import collections
from tkinter.filedialog import askopenfilename

#labels which one wants to extract
extractionLabels = ["exemplification","exception","explain","specification","justify","motivate","cause","purpose","reason","contrast"]

corpus = []

def readFile(path):
    file = open(path)
    lines = [line.strip().split('\t') for line in file]
    return lines

def print2File(filename):
    if os.path.isfile('./'+filename):
        file = open('./'+filename, 'a')
        for elem in corpus:
            file.write(elem[2]+"\t"+elem[0]+"\t"+elem[1]+"\n")
    else:
        file = open('./'+filename, 'w+')
        for elem in corpus:
            file.write(elem[2]+"\t"+elem[0]+"\t"+elem[1]+"\n")

datapathOne = askopenfilename(initialdir= os.getcwd(), title="Select original file")
original_file = readFile(datapathOne)

datapathTwo = askopenfilename(initialdir= os.getcwd(), title="Select file with potential duplicates")
duplicate_file = readFile(datapathTwo)
old_length = len(duplicate_file)

lineCounter = -1
minLength = 8
maxLength = 25
tooShort = True
tooLong = False

singletons = []

for duplicate in duplicate_file:
    lineCounter += 1
    kicked = False

    #too short
    if tooShort:
        if len(duplicate[1].strip().split()) <= minLength or len(duplicate[2].strip().split()) <= minLength:
            #print(duplicate_file[lineCounter])
            #del duplicate_file[lineCounter]
            continue

    #too long
    if tooLong:
        if len(duplicate[1].strip().split()) >= maxLength or len(duplicate[2].strip().split()) >= maxLength:
            #print(duplicate_file[lineCounter])
            #del duplicate_file[lineCounter]
            continue

    #check for duplicates
    for original in original_file:
        if duplicate[2] == original[1] or duplicate[2] == original[2]:

            #del duplicate_file[lineCounter]
            kicked = True
            continue
            #print("Deleted duplicate")

        if duplicate[1] == original[1] or duplicate[2] == original[2]:

            #del duplicate_file[lineCounter]
            kicked = True
            #print("Deleted duplicate")
    if not kicked:
        singletons.append(duplicate)




file = open('./'+"singletons.txt", 'a')
for elem in singletons:
    file.write(elem[0]+"\t"+elem[1]+"\t"+elem[2]+"\n")

def average(data):

    min = 100
    max = 0
    maxLine = 0
    wordCount = 0
    lineNumber = -1
    for line in data:
        lineNumber += 1
        wordCount += len(line[1].strip().split())
        wordCount += len(line[2].strip().split())
        if len(line[1].strip().split()) > max :
            max = len(line[1].strip().split())
            maxLine = lineNumber
        if len(line[1].strip().split()) < min:
            min = len(line[1].strip().split())
        if len(line[1].strip().split()) < min:
            min = len(line[1].strip().split())
        if len(line[2].strip().split()) > max :
            max = len(line[2].strip().split())
            maxLine = lineNumber
        if len(line[2].strip().split()) < min:
            min = len(line[2].strip().split())


    print("Words: "+ str(wordCount))
    print("Average: "+ str(wordCount/(len(data)*2)))
    print("Max: "+ str(max))
    print("Min: "+ str(min))
    print(data[maxLine])


average(original_file)
print("XXXXXXXXXX")
average(singletons)

print(old_length)
print(len(singletons))