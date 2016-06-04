import re, os, collections
from tkinter.filedialog import askopenfilename


def convertTimes(times):
    print(times)
    startMin = int(times[0][0])
    endMin = int(times[1][0])
    startSec = int(times[0][1])
    endSec = int(times[1][1])

    #for minutes.seconds
    #converted = [float(startMin*60+startSec),float(endMin*60+endSec)]
    #for seconds.milliseconds
    converted = [float(str(startMin)+"."+str(startSec)),float(str(endMin)+"."+str(endSec))]

    return converted


def buildElements(file):
    elems = collections.defaultdict(str)
    index = collections.defaultdict(int)
    file= open(file)

    for line in file:
        line = line.split(' ', 1)
        #time = re.findall( r'(\d{2})\.+\d', line[0])
        time = re.findall( r'(\d{2,4})\.(\d{1,4})', line[0])
        speaker = str(re.findall(r'\d_', line[0]))

        time = convertTimes(time)
        elems[speaker] += "<el index=\"" + str(index[speaker]) + "\" " + "start=\"" + str(time[0]) + "\" " + "end=\"" + str(time[1]) + "\">" +"\n"
        index[speaker] += 1
        elems[speaker] += "<attribute name=\"token\">"+line[-1]+"</attribute>" +"\n"
        elems[speaker] += "</el>" + "\n"

    return list(elems.values())


def main():
    #mergedFiles = False if every speaker has its on ASR file
    mergedFiles = True

    #Speaker denotation for specification
    proposer = "participantA"
    opponent = "participantB"

    #data paths
    relPath = os.getcwd()
    pathSpec = r"E:\Thesis\data\SyncronisedAudioMetalogue\diaml-spec-v0.6.xml"
    pathVid = r"E:\Thesis\data\SyncronisedAudioMetalogue\Overall\SyncronisedAudioOverall48kHz24bit-prepilot-athensFeb15-FFF-Session-004-MMM.avi"
    pathASR = r"E:\Thesis\data\SyncronisedAudioMetalogue\ASR"

    #xml fragments
    header ="<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>" + "\n"\
            "<annotation>" + "\n"\
                "<head>" + "\n"\
                "<specification src=\"" + pathSpec +"\" />" +"\n"\
                "<video src=\"" + pathVid + "\" />" + "\n"\
                "</head>" + "\n"\
                "<body>" +"\n"\
                "<track name=\""+proposer+".utterance\" type=\"primary\">" +"\n"
    changeTrack = "</track>"+ "\n"\
            "<track name=\""+opponent+".utterance\" type=\"primary\">" + "\n"\

    tail =  "</track>" +"\n"\
            "</body>" +"\n"\
            "</annotation>"

    Speaker =askopenfilename(initialdir=relPath, title="Select ASR file")
    elems = buildElements(Speaker)


    if not mergedFiles:
        Speaker2 = askopenfilename(initialdir=relPath, title="Select ASR file for Speaker 2")
        elems[1] = buildElements(Speaker2)



    #write to file
    f = open('test.anvil','w')
    f.write(header+elems[0]+changeTrack+elems[1]+tail)
    f.close()

if __name__ == "__main__":
    main()