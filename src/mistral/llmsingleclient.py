import json
import tqdm
import requests
import sys
import re

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
  text = re.sub("</s>", "", struct["data"]["predictions"][0])

  if verbose:
    print(text)
    print("----------")

  return text

