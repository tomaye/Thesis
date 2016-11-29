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
        bar = progressbar.ProgressBar(max_value=len(text))
        X = []

        total = 0

        for i in range(0, len(text)):
            temp =[]
            for sent in text[i]:
                f = open(input, "w+", encoding="utf-8")
                f.write(sent)
                f.close()
                subprocess.run([senna+cmd], shell=True)
                f = open(output, "r", encoding="utf-8")

                counter = 0
                for line in f:
                    line = line.split("\t")
                    #print(line)
                    if line != ["\n"]:
                        if "B" in line[1]:
                            counter += 1
                f.close()
                temp.append(counter)
            X.append(temp)
            bar.update(i+1)

        subprocess.run("rm "+input, shell=True)
        subprocess.run("rm "+output, shell=True)

        return np.array(X)

    def count_args(self, text):
        tag = " -srl"
        cmd = setpath + tag + " < " + input + " > " + output
        bar = progressbar.ProgressBar(max_value=len(text))
        X = []

        for i in range(0, len(text)):
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
                            counterMods += 1
                        elif "A0" in line[-1] or "A1" in line[-1] or "A2" in line[-1]:
                            counterArgs += 1
                f.close()
                temp.append(counterArgs)
                temp.append(counterMods)
            X.append(temp)
            bar.update(i+1)

        subprocess.run("rm "+input, shell=True)
        subprocess.run("rm "+output, shell=True)

        return np.array(X)

    def count_constituents(self, text):
        '''
        counts syntactic constituents
        :param text:
        :return:
        '''
        tag = " -psg"
        cmd = setpath + tag + " < " + input + " > " + output
        bar = progressbar.ProgressBar(max_value=len(text))
        X = []

        for i in range(0, len(text)):
            sent = text[i]
            temp =[]
            f = open(input, "w+", encoding="utf-8")
            f.write(sent)
            f.close()
            subprocess.run([senna+cmd], shell=True)
            f = open(output, "r", encoding="utf-8")

            counter = 0
            for line in f:
                line = line.split("\t")
                #print(line)
                if line != ["\n"]:
                    counter += line[-1].count("(")
            f.close()
            temp.append(counter)
            X.append(temp)
            bar.update(i+1)

        subprocess.run("rm "+input, shell=True)
        subprocess.run("rm "+output, shell=True)
        return np.array(X)


    def save_as_file(self, text, tag):

        if tag == "srl":

            X = self.count_args(text)
            f = open("srl.txt", "w+", encoding="utf-8")

            for i in range(0,len(text)):
                f.write(str(text[i])+"\t"+str(X[i])+"\n")

        if tag == "chk":

            X = self.count_chunks(text)
            f = open("chk.txt", "w+", encoding="utf-8")

            for i in range(0,len(text)):
                f.write(str(text[i])+"\t"+str(X[i])+"\n")

    def load_from_file(self, text, tag):

        if tag == "srl":
            f = open("srl.txt", encoding="utf-8")

        elif tag == "chk":
            f = open("chk.txt", encoding="utf-8")

        dic = {}

        for line in f:
            [key, value] =line.split("\t")
            dic[key] = value

        X = []
        for sentences in text:
            numbers = dic[str(sentences)].replace("\n","").replace("[", "").replace("]", "")
            ints = [int(i) for i in numbers.split()]
            #single = sum(ints)
            #X.append([single])
            X.append(ints)

        return np.array(X)
