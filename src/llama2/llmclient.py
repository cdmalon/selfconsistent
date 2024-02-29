import json
import tqdm
import requests
import sys
import re
import nltk

def camel(prompts, verbose=False):
  secret = "llama2"

  if(len(prompts)) != 1:
    raise ValueError("Don't support batches yet")

  prompt = prompts[0]
  if verbose:
    print("PROMPT: " + prompt)

  senddata = {"secret": secret, "prompts": prompts}
  jsonsend = json.dumps(senddata)
  
  headers={'Content-Type':'application/json; charset=utf-8'}
  r = requests.post('http://127.0.0.1:6053/predict', json=senddata, headers=headers)
  struct = json.loads(r.text)
  n = len(prompt)
  generations = []
  lastsent = False
  for text in struct["data"]["predictions"]:
    remainder = text[n+4:]
    stripped = re.sub("</s>", "", remainder)
    if(not re.search(r"[a-zA-Z]", stripped)):
      lastsent = True
      continue
    sents = nltk.tokenize.sent_tokenize(stripped)
    if(len(sents) == 0):
      lastsent = True
      continue
    if(len(sents) == 1):
      lastsent = True

    generations.append(sents[0])

    if verbose:
      print(lastsent, sents[0])

  if verbose:
    print("----------")
  return generations, lastsent

