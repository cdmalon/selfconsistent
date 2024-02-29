import json
import tqdm
import sys
import yaml
import re
import spacy
import nltk.tokenize

nlp = spacy.load("en_core_web_sm")

class IndependentDecoder():
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
  
    choice = 0
    best = 0
    for i, r in enumerate(responses):
      if(scores[i] > best):
        best = scores[i]
        choice = i
  
    return choice


  def decode(self, base_prompt):
    prompt = base_prompt
 
    generations = self.api([prompt])

    groups = []
    is_last = []
    for stripped in generations:
      sents = nltk.tokenize.sent_tokenize(stripped)
      for i, sent in enumerate(sents):
        if(i == len(groups)):
          groups.append([])
          is_last.append([])
  
        groups[i].append(sent)
  
        if(i+1 == len(sents)):
          is_last[i].append(True)
        else:
          is_last[i].append(False)
  
    for j, group in enumerate(groups):
      if(len(group)) == 0:
        break
      else:
        if(not re.search(r"[a-zA-Z]", group[0])):
          if(not re.search(r"^\s*\d+\.\s*$", group[0])):
            break
  
        generalization_idx = self.choose_generalization(group, group)
        generalization = group[generalization_idx]
 
        if self.verbose: 
          print("GENERALIZE: " + generalization)

        prompt = prompt + " " + generalization + " "
  
        if is_last[j][generalization_idx]:
          break
  
    n = len(base_prompt)
    result = prompt[n:]

    return result

