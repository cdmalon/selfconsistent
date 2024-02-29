import json
import tqdm
import sys
import yaml
import re
import spacy

nlp = spacy.load("en_core_web_sm")

class SelfConsistentDecoder():
  def  __init__(self, api=None, verbose=False):
    self.api = api
    self.verbose = verbose

  def grammar_ok(self, text):
    space = nlp(text)
  
    found_subj = False
    found_verb = False
  
    for sent in space.sents:
      for tok in sent:
        if tok.dep_ == "nsubj" or tok.dep_ == "nsubjpass" or tok.dep_ == "expl":
          found_subj = True
        if tok.tag_ == "VBZ" or tok.tag_ == "VBD" or tok.tag_ == "VBP" or tok.dep_ == "aux" or tok.dep_ == "auxpass":
          found_verb = True
  
    if found_subj and found_verb:
      return True

    if(re.search(r"^\s*\d+\.\s*$", text)):
      return True   # Allow ordered lists
  
    return False
  
  def choose_generalization(self, premises, responses):
    premise_words = []
    for p in premises:
      words = {}
      for w in p.split():
        words[w.lower()] = 1
      premise_words.append(words)
  
    scores = []
    for response in responses:
      words = response.split()
      n = 0
      s = 0
      for word in words:
        word = word.lower()
        n = n + 1
        for p in premise_words:
          if word in p:
            s = s + 1
  
      if(re.search(r"\|", response) or self.grammar_ok(response) == False):
        scores.append(0)
        if self.verbose:
          print("REJ", response)
      elif(n > 0):
        scores.append(s/n)
        if self.verbose:
          print("GEN", str(s/n), response)
      else:
        scores.append(0)
  
    choice = None
    best = 0
    for i, r in enumerate(responses):
      if(scores[i] > best):
        best = scores[i]
        choice = r
  
    if best == 0:
      choice = premises[0]
  
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
          if(not re.search(r"^\s*\d+\.\s*$", generations[0])):
            break
  
        generalization = self.choose_generalization(generations, generations)
 
        if self.verbose: 
          print("GENERALIZE: " + generalization)
        prompt = prompt + " " + generalization + " "
        last_generation = generalization
        
    n = len(base_prompt)
    result = prompt[n:]
    return result
  
