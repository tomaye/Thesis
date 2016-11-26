from collections import defaultdict
import statistics
from features import token_counter, chunk_counter

file = "data/CompQuestResults.txt"
#1 column data 19 results


f = open(file, encoding="utf-8")

survey = defaultdict(int)
text = []

i = 1
for line in f:
    line = line.split("\t")
    text.append(line[0])

    #delete \n
    line[19] = line[19][:-1]

    dic = {}
    dic["text"] = line[0]
    dic["raw"] = line[1:]
    dic["median"] = statistics.median(sorted(line[1:]))
    survey[i] = dic
    i += 1

senna = chunk_counter.ChunkcountVectorizer()

senna.count_psg(text)
