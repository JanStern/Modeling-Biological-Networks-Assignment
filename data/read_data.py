# Modelin Biological Networks Assignment
#
# Jan Sternagel, 102177941

import json

with open("data.txt", "r") as f:
    data = f.readlines()


gene_names = data[0].replace("\n", "").split(" ")

res = {gen: [] for gen in gene_names}

for l in data[1:]:
    values = l.replace("\n", "").split(" ")
    
    for i_gen, val in enumerate(values):
        res[gene_names[i_gen]].append(float(val))
    
res["t"] = list(range(0, 200, 10))

with open("data_original.json", "w") as f:
    json.dump(res, f, indent=4)