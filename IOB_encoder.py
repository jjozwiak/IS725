import json
import nltk
import csv
from nltk.tokenize import sent_tokenize, word_tokenize


# load annotation data
with open('data/is-725-training-data-annotation_annotations.json') as f:
    data = json.load(f)


example_index = 7

print(len(data["examples"]))

for i in range(len(data["examples"])):

    content = data["examples"][i]["content"]
    annotations = data["examples"][i]["annotations"]


    # Build the tags dictionary
    tagsDict = {}

    for annotation in annotations:
        tag = annotation["tag"]
        tokens = nltk.word_tokenize(annotation["value"].lower())
        for index, token in enumerate(tokens, start=0):
            if (index == 0):
                tagsDict[token.lower()] = 'B_' + tag
            else:
                tagsDict[token.lower()] = 'I_' + tag


    tokenized = nltk.word_tokenize(content)

    # Dictionary to store the output for our encoding
    outputDict = {}

    for index, token in enumerate(tokenized, start=0):
        outputDict[index] = {'word': token, 'pos': '', 'iob': ''}

    for index, item in enumerate(outputDict, start=0):
        current_word = outputDict[index]["word"].lower()
        if (current_word in tagsDict.keys()):
            outputDict[index]["iob"] = tagsDict[current_word]
        else:
            outputDict[index]["iob"] = 'O'


    def tag_pos():
        for i in tokenized:
            tagged = nltk.pos_tag(tokenized)

        taggedOutput = {}
        for index, item in enumerate(tagged, start=0):
            taggedOutput[index] = {'word': item[0], 'pos': item[1]}

        return taggedOutput


    tagged_data = tag_pos()


    # Merge IOB tags with POS tags
    for index, item in enumerate(tagged_data, start=0):
        outputDict[index]["pos"] = tagged_data[index]["pos"]

    #file_output_name = 'data/iob_output/proj_output_' + str(i) + '.csv'
    file_output_name = 'data/iob_output/proj_output.csv'


    with open(file_output_name, mode='a', newline='') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for index, item in enumerate(outputDict, start=0):
            employee_writer.writerow([outputDict[index]["word"], outputDict[index]["pos"], outputDict[index]["iob"]])