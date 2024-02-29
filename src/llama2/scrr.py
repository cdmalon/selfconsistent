import json
import tqdm
import sys
import yaml
import re
import nltk.tokenize
from summac.model_summac import SummaCZS

class SCRRDecoder():
  def  __init__(self, api=None, verbose=False):
    self.api = api
    self.verbose = verbose
    self.model = SummaCZS(granularity="document", model_name="anli", bins="percentile", use_con=False, device="cuda")


  def choose_generalization(self, generations):
    scores = []
    for i in range(len(generations)):
      tot = 0
      for j in range(len(generations)):
        e = self.model.score([generations[j]], [generations[i]])["scores"][0]
        tot = tot + e
      avg = tot/len(generations)
      if self.verbose:
        print(avg, generations[i])
      scores.append(avg)
  
    best = sorted(range(len(generations)), key=lambda x: scores[x], reverse=True)[0]
  
    result = generations[best]
    sents = nltk.tokenize.sent_tokenize(result)
    finalresult = []
    for sent in sents:
      if(not re.search(r"[a-zA-Z]", sent)):
        break
      finalresult.append(sent)
    result = " ".join(finalresult)
    if self.verbose:
      print("GENERALIZE: ", result)
  
    return result


  def decode(self, base_prompt):
    lastsent = False
    prompt = base_prompt
 
    generations, lastsent = self.api([prompt])

    result = self.choose_generalization(generations)
 
    return result

