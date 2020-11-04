"""
    Read file with CONLL-U format and prepare for POS-Tagging
"""


conllu_file = open("/Users/leo/Work/Universal Dependencies 2.5/French/UD_French-Sequoia/fr_sequoia-ud-dev.conllu")
for line in conllu_file.readlines()[:100]:
    #print(line)
    if len(line) < 2 or line[0] == "#":
        continue
    annotation = line.split("\t")
    #print(annotation)
    word = annotation[1]
    label = annotation[4]
    attributes = annotation[6]
    print(word, label, attributes)
