from collections import defaultdict
import statistics
from features import token_counter, chunk_counter

file = "data/CompQuestResults.txt"
#1. column data 19 results


f = open(file, encoding="utf-8")

survey = defaultdict(int)
text = []
medians = []

i = 1
for line in f:
    line = line.split("\t")
    text.append(line[0])

    #delete \n
    line[19] = line[19][:-1]

    dic = {}
    dic["text"] = line[0]
    dic["raw"] = line[1:]
    median = statistics.median(sorted(line[1:]))
    dic["median"] = median
    medians.append(median)
    survey[i] = dic
    i += 1

senna = chunk_counter.ChunkcountVectorizer()

syn = senna.count_constituents(text)
sem = senna.count_args([[sent,""] for sent in text])
length = [len(sent.split()) for sent in text]

print("\n")
print("syntactic constituents: ")
print([int(row) for row in syn])
print("semantic constituents: ")
print([sum(row)for row in sem])
print("length: ")
print(length)
print("median: ")
print([int(median) for median in medians])

