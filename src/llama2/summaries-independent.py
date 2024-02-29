import json
import tqdm
import sys
import re

import llmindclient
import independent

in_fp = open(sys.argv[1], "r")
all = in_fp.read()
struct = json.loads(all)
in_fp.close()

decoder = independent.IndependentDecoder(api=llmindclient.camel, verbose=False)

done = {}
for example in tqdm.tqdm(struct):
  hash = example["hash"]
  if hash in done:
    continue
  done[hash] = 1

  document = example["article"]
  document = re.sub(r"\.([a-zA-Z])", ". \\1", document)
  document = re.sub(r"([a-z])([A-Z])", "\\1 \\2", document)
  document = re.sub(r"Share this with Email Facebook Messenger Messenger Twitter Pinterest Whats App Linked In Copy this link", "", document)

  base_prompt = "[INST] Summarize the following article.\nArticle: " + document + "\nSummary: [/INST] "

  prediction =  decoder.decode(base_prompt)

  print(json.dumps({"article": document, "hash": hash, "summary": prediction}))

