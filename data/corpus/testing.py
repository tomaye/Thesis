'''

f =open("IBM_extracted_raw.txt", encoding="utf-8")
#f =open("IBM_extracted_raw_negatives.txt", encoding="utf-8")
f2 =open("IBM_test_neg.txt","w+", encoding="utf-8")

#f2 =open("IBM_extracted_raw.txt","w+", encoding="utf-8")

for line in f:
    line = line.replace("“","").replace("’","").replace("—","-").replace("€","Euro ").replace("–","").replace("é","e").replace("”","").replace("‘","").replace("…","").replace("ü","u")
    line = line.replace("ó","o").replace("É","E").replace("ï","i").replace("á","a").replace("ó","o").replace("è","e").replace("ò","o").replace("ñ","n").replace("­","").replace("î","i")
    line = line.replace("í","i").replace("å","a").replace("ç","c").replace("§","").replace("ä","a").replace("ö","o").replace("£","").replace("†","").replace("Ó","O").replace("°","").replace("Á","A")
    line = line.replace("æ","ae").replace(" "," ").replace("ô","o").replace("â","a").replace("¢","").replace("‡","").replace("Ž","Z").replace("ž","z").replace("ø","o").replace("ë","e")
    line = line.replace("ª","").replace("š","s").replace("ú","u").replace("·","").replace("ã","a").replace("Ü","U").replace("û","u").replace("à","a").replace("ê","e").replace("Å","A")
    line = line.replace("Ç","C").replace("Ú","U").replace("³","").replace("²","").replace("¥","").replace("ß","ss").replace("−","").replace("ð","d").replace("Ö","O").replace("ı","")
    line = line.replace("Ḥ","H").replace("ć","c").replace("ř","r").replace("ğ","g").replace("İ","I").replace("ğ","g")
    f2.write(line)ascii
    #line= line.split("\t")
    #f2.write(str(line[0].encode('ascii', 'replace'))+"\t"+str(line[1].encode('ascii', 'replace'))+"\t"+str(line[2].encode('ascii', 'replace'))+"\n")

'''

f = open("../../srl.txt", encoding="utf-8")

i = 0
for line in f:
    line = line.split("\t")
    if line[1] != None:
        i += 1

print(i)