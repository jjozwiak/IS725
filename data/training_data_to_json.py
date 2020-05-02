import os
import json

directory = 'prs/training/trimmed'

output = []

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        f = open(os.path.join(directory, filename))
        lines = f.read()
        output.append({'message': lines})
        f.close()

output_file = open(os.path.join(directory, 'output.json'), 'w')
output_file.write(json.dumps(output))
output_file.close()