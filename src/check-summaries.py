import json
import tqdm
import sys
import re
import os
import nltk.tokenize
from summac.model_summac import SummaCZS

in_file = sys.argv[1]

model = SummaCZS(granularity="sentence", model_name="vitc", bins="percentile", use_con=False, device="cuda")

in_fp = open(in_file, "r")

tot = 0
n = 0

for line in tqdm.tqdm(in_fp.readlines()):
  if(re.search(r"^{", line)):
    example = json.loads(line)
    document = example["article"]
    summary = example["summary"]

    score = model.score([document], [summary])["scores"][0]
    tot = tot + score
    n = n + 1

avg = tot/n
print("Average score: " + str(avg) + " (" + str(n) + ")")

