import json
import tqdm
import sys
import yaml
import re
import nltk.tokenize
from selfcheckgpt.modeling_selfcheck import SelfCheckNgram

class NGramDecoder():
  def  __init__(self, api=None, verbose=False):
    self.api = api
    self.verbose = verbose
    self.selfcheck_ngram = SelfCheckNgram(n=1)


  def choose_generalization(self, responses):
    scores = []
    for i, response in enumerate(responses):
      others = responses[:i] + responses[i+1:]
      s = self.selfcheck_ngram.predict(passage=response, sentences=[response], sampled_passages=others)["sent_level"]["max_neg_logprob"][0]
      scores.append(s)
  
    choice = responses[0]
    best = scores[0]
    for i, r in enumerate(responses):
      if(scores[i] < best):
        best = scores[i]
        choice = r
  
    return choice


  def decode(self, base_prompt):
    lastsent = False
    prompt = base_prompt
 
    last_generation = None
    while lastsent == False:
  
      generations, lastsent = self.api([prompt])
  
      if(len(generations)) == 0:
        break
      else:
        if(not re.search(r"[a-zA-Z]", generations[0])):
          break
  
        generalization = self.choose_generalization(generations)
 
        if self.verbose: 
          print("GENERALIZE: " + generalization)
        prompt = prompt + " " + generalization + " "
        last_generation = generalization
  
    n = len(base_prompt)
    result = prompt[n:]
 
    return result

