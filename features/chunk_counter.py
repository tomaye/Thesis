import subprocess
import numpy as np

input = "./senna_input.txt"
output = "./senna_output.txt"

senna = "/home/frigori/Senna/senna/senna-linux64"
setpath = " -path /home/frigori/Senna/senna/"
tag = " -chk"
cmd = setpath + tag + " < " + input + " > " + output

test = ["I want some food and like playing golf","JUST TESTING SOMETHING HERE"]



class ChunkcountVectorizer():

    def count_chunks(self, text):

        X = []

        for sent in text:
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
            X.append(counter)

        subprocess.run("rm "+input, shell=True)
        subprocess.run("rm "+output, shell=True)

        return np.array(X)
#TODO
sentPairwise array


