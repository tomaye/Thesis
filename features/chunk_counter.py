import subprocess
import numpy as np

input = "./senna_input.txt"
output = "./senna_output.txt"

senna = "/home/frigori/Senna/senna/senna-linux64"
setpath = " -path /home/frigori/Senna/senna/"
tag = " -chk"
#Semantic Role Label tagging
tag = " -srl"
cmd = setpath + tag + " < " + input + " > " + output

test = [["I want some food and like playing golf", "JUST TESTING SOMETHING HERE"], ["this is sentence two", "my party stands firmly convinced"]]



class ChunkcountVectorizer():

    def count_chunks(self, text):

        X = []

        total = 0

        for sentpair in text:
            temp =[]
            for sent in sentpair:
                f = open(input, "w+", encoding="utf-8")
                f.write(sent)
                f.close()
                subprocess.run([senna+cmd], shell=True)
                f = open(output,"r", encoding="utf-8")

                counter = 0
                for line in f:
                    line = line.split("\t")
                    print(line)
                    if line != ["\n"]:
                        if "B" in line[1]:
                            counter += 1
                f.close()
                temp.append(counter)
            X.append(temp)
            total += 1
            print(str(total) + "/" + str(len(text)))
        subprocess.run("rm "+input, shell=True)
        subprocess.run("rm "+output, shell=True)

        #print(np.array(X))
        return np.array(X)

vec = ChunkcountVectorizer()
vec.count_chunks(test)
