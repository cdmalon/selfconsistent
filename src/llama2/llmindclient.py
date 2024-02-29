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

  for text in struct["data"]["predictions"]:
    remainder = text[n+4:]
    stripped = re.sub("</s>", "", remainder)
    generations.append(stripped)

    if verbose:
      print(stripped)

  if verbose:
    print("----------")

  return generations

