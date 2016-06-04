import os
from tkinter.filedialog import askopenfilename

file = askopenfilename(initialdir=os.getcwd(), title="Select file for cleaning")
fileIn= open(file)
fileOut = open(file+"_cleaned.txt", 'w+')

switchColumns = False

changeLabels = True
justification = ["justification", "reason", "justify", "motivate", "cause"]
evidence = ["explain","exemplification","exemplify", "exception"]
negative = ["contrast", "elaboration"]

for str in fileIn:

    str = str.strip()
    str = str.replace("</connective>", "")
    str = str.replace("<connective>", "")
    str = str.replace("_", " ")
    str = str.replace("[", "")
    str = str.replace("]", "")

    if switchColumns:
        line = str.split("\t")
        line = line[2]+"\t"+line[0]+"\t"+line[1]+"\n"

    if changeLabels:
        line = str.split("\t")
        if line[0] in justification:
            line[0] = "justification"
        elif line[0] in evidence:
            line[0] = "evidence"
        elif line[0] in negative:
            line[0] = "negative"
        line = line[0]+"\t"+line[1]+"\t"+line[2]+"\n"


    fileOut.write(line)

print("DONE")