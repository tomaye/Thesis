import os, collections

relPath = "data/corpus/"

def getStats(data):
    labelCount = collections.defaultdict(int)
    totalCount = 0

    #print(relPath+data)
    file = open(relPath+data)
    for line in file:
        line = line.split("\t")
        labelCount[line[0]] += 1
        totalCount += 1

    print("XXXXXXXXXXXXXXXXXXXXX"+"\n"+str(totalCount)+" labels found in "+data)
    for label in labelCount:
        print(label+" : "+str(labelCount[label])+" Percentage: "+str(round((labelCount[label]/totalCount), 2)))
    print("XXXXXXXXXXXXXXXXXXXXX"+"\n")

def main():
    for file in os.listdir(os.getcwd()+"/"+relPath):
        if file.endswith(".txt"):
            getStats(file)

if __name__ == "__main__":
    main()
    print("testing")
    print("DONE")
