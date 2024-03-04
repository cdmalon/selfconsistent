# Self Consistent Decoding

This repository supplements the paper, "[Self-Consistent Decoding for
More Factual Open Responses](https://arxiv.org/abs/2403.00696)," by Christopher Malon and Xiaodan Zhu.

## Requirements

Our complete Conda environment is given in `env.yml`, which is not minimal.

```
conda env create -n llama2 -f env.yml
conda activate llama2
python -m spacy download en_core_web_sm
```

If you can already run your Llama 2 model (or other LLM) under Huggingface
Transformers, the main additional dependencies are `requests`,
`nltk`, and `spacy`, but `nltk` and `spacy` are needed only for
English-specific sentence splitting and filters.

For the Mistral servers, use `mistral-env.yml`.

## Serve your language model

Our decoding script expects access to a language model via HTTP.
The script `flask_2048_sampling_13b.py` can be used to serve five sampled
sequences from your model.  Change the `model_name` to the path
of your HuggingFace model.

The Flask server script is invoked with a command like:

```
waitress-serve --host 127.0.0.1 --port 6053 --call flask_2048_sampling_13b:web
```

## Run the decoder

The decoder is implemented in `selfconsistent.py` and the LLM client is
implemented in `llmclient.py`.  As an example to generate summaries for
`human_annotations_sentence.json` from
[FRANK](https://github.com/artidoro/frank/tree/main/data),
where the input is given as a JSON file with a list of examples including
"hash" and "article", you can run

```
python summaries-selfconsistent.py human_annotations_sentence.json
```

assuming that you have the LLM server running on the localhost.
This will output a JSON file containing a list of "article", "hash",
and "summary", which is the newly generated summary.

## Confirming SummaC scores

The script `check-summaries.py` will compute SummaC ZS score of
an output file.

## Supporting non-English languages

To support a non-English language, minimally, an appropriate sentence
splitter needs to be substituted for `nltk.tokenize.sent_tokenize()`.
Regular expression checks that assume that alphanumeric characters will occur
in valid output should be replaced by appropriate character sets.
The `grammar_ok()` function, which checks an English sentence for the existence
of a subject and verb, can either be disabled (assume True) or by rules
appropriate to the target language.

Spacy will no longer be required if you replace `grammar_ok()`,
and `nltk` will no longer be required if you replace the sentence splitter.

## Optimization

As future work, the call to `model.generate()` in `flask_2048_sampling_13b.py`
should be replaced by a function that stops decoding as soon as one sentence
is decoded in each sample, instead of continuing to the end of the response.
This will avoid time spent generating tokens that are thrown away and
replaced in the sampling of the next sentence.

## Baseline configurations

Here are the script combinations to use in place of 
`flask_2048_sampling_13b.py` and `summaries-selfconsistent.py` for
the other baselines:

| **Model** | **Server** | **Script** |
|--------------|------------|------------|
| Greedy | flask_2048_greedy_13b.py | summaries-one.py |
| Nucleus | flask_2048_nuc_13b.py | summaries-one.py |
| Beam | flask_2048_beam_13b.py | summaries-one.py |
| P-CRR | flask_2048_pcrr_13b.py | summaries-one.py |
| S-CRR | flask_2048_sampling_13b.py | summaries-scrr.py |
| DoLa | flask_2048_dola_13b.py | summaries-one.py |
| SelfCheckGPT | flask_2048_sampling_13b.py | summaries-selfcheckgpt.py |
| Independent | flask_2048_sampling_13b.py | summaries-independent.py |
| Sample & Select | flask_2048_sampling_13b.py | summaries-selfconsistent.py |

Note that the DoLa server requires a specially modified version of
transformers providing the required classes, which is distributed
with [DoLa](https://github.com/voidism/DoLa).  Put this version of
transformers in your PYTHONPATH when running the server.  

## Outputs

Outputs from each system on the test subset of FRANK,
as reported in the paper, are in the outputs subdirectory.

If you wish to split the outputs into CNN/DM and XSum components,
you may run
```
perl split-by-hash.pl all-outputs.jsonl cnn-outputs.jsonl xsum-outputs.jsonl
```
where `all-outputs.jsonl` is the merged output file and the other two
files are to be created.

## Human evaluations

In the JSONL file in the human-eval subdirectory, each line represents
one system's outputs on one of the 50 articles.  The index is the
hash of the article from CNN/DM or XSum, followed by a colon, followed
by a code indicating the name of the system:

| **Code** | **System** |
|-------|-----------|
| beam5 | Beam |
| chkngram6 | SelfCheckGPT |
| ind6 | Independent |
| sel6 | Sample & Select |

If the code is different from these four, the summary was an alertness test
from the original FRANK dataset, and the code refers to one of the
systems tested in that paper.

The lists "summac_zs", "qafe", and "human" are in one-to-one correspondence
with the "sentences" of the summary.  Each item of "human" gives
the factuality classifications by the three workers.

