import subprocess
import numpy as np
import progressbar, time

input = "./senna_input.txt"
output = "./senna_output.txt"

senna = "/home/frigori/Senna/senna/senna-linux64"
setpath = " -path /home/frigori/Senna/senna/"
tag = " -chk"
#Semantic Role Label tagging
tag = " -srl"
cmd = setpath + tag + " < " + input + " > " + output

test = [["I want some food and like playing golf", "JUST TESTING SOMETHING HERE"], ["this is not sentence two", "my party does not stand firmly convinced"]]



class ChunkcountVectorizer():

    def count_chunks(self, text):
        tag = " -chk"
        cmd = setpath + tag + " < " + input + " > " + output

        X = []

        total = 0

        for sentpair in text:
            temp =[]
            for sent in sentpair:
                f = open(input, "w+", encoding="utf-8")
                f.write(sent)
                f.close()
                subprocess.run([senna+cmd], shell=True)
                f = open(output, "r", encoding="utf-8")

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

    def count_args(self, text):
        tag = " -srl"
        cmd = setpath + tag + " < " + input + " > " + output
        bar = progressbar.ProgressBar(max_value=len(text))
        X = []

        for i in range(0,len(text)):
            temp =[]
            for sent in text[i]:
                f = open(input, "w+", encoding="utf-8")
                f.write(sent)
                f.close()
                subprocess.run([senna+cmd], shell=True)
                f = open(output, "r", encoding="utf-8")

                counterArgs = 0
                counterMods = 0
                for line in f:
                    line = line.split("\t")
                    #print(line)
                    if line != ["\n"]:
                        if "AM-MOD" in line[-1] or "AM-MNR" in line[-1]:
                            counterArgs += 1
                        elif "A0" in line[-1] or "A1" in line[-1] or "A2" in line[-1]:
                            counterMods += 1
                f.close()
                temp.append(counterArgs)
                temp.append(counterMods)
            X.append(temp)
            bar.update(i)

        subprocess.run("rm "+input, shell=True)
        subprocess.run("rm "+output, shell=True)

        return np.array(X)


#vec = ChunkcountVectorizer()
#vec.count_args(test)
