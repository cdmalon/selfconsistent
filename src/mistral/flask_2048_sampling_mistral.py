import json
import sys
import os
import tqdm
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
import logging
from logging.config import dictConfig

from flask import Flask, request
from flask_responses import json_response

model_name = "/mnt/ml_data/malon/mistral/Mistral-7B-Instruct-v0.2"
info = model_name.split("/")
version = info[-1]

max_length = 2048
batch_size = 4

torch.cuda.manual_seed(42)
torch.manual_seed(42)

secret = "llama2"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {   
        "eos_token": "</s>",
        "bos_token": "</s>",
        "unk_token": "</s>",
        "pad_token": "[PAD]",
    }
)


def wrap_message(status,data):
    return {"result":status,"data":data}


def wrap_error(message):
    return wrap_message("error", {"message":message})

def wrap_predictions(predictions, version):
    return wrap_message("success", {"version": version, "predictions":predictions})


def web():
  app = Flask(__name__)
  app.logger.info("Init LLM server")

  @app.route("/predict", methods=["POST"])
  def predict():
    if request.json is None:
      return json_response(wrap_error("Expected a JSON request"), 400)

    if "secret" not in request.json or request.json["secret"] != secret:
      return json_response(wrap_error("Invalid authentication"), 403)

    if "prompts" not in request.json or not isinstance(request.json["prompts"], list):
      return json_response(wrap_error("No prompts"), 400)

    statements = request.json["prompts"]
    for i, x in enumerate(statements):
      statements[i] = "</s>" + x

    all_responses = []

    if(len(statements) != 1):
      raise ValueError("Only one statement supported for now")

    try:
      while(len(statements) > 0):
        batch = statements[:batch_size]
        statements = statements[batch_size:]

        encoded = tokenizer.batch_encode_plus(batch, add_special_tokens=False, return_tensors="pt", truncation=True, padding=False).to("cuda:0")
        max_input = encoded["input_ids"].shape[1]
        inputs = {}
        for k in encoded:
          if(k != "token_type_ids"):
            inputs[k] = encoded[k]
        outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=5, do_sample=True, top_p=.9).to("cpu") # temperature, top_k, top_p, repetition_penalty
        # outputs = outputs[:, max_input:]
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=False)
      
        for i in range(len(responses)):
          response = responses[i]
          all_responses.append(response)
          # print(response)

    except Exception as e:
      return json_response(wrap_error(str(e)), 500)

    return json_response(wrap_predictions(all_responses, version))

  return app

