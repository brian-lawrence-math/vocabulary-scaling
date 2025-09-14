# vocabulary-scaling
An experiment in scaling ML models to learn vocabulary items

Background: While training a machine translation (English-Spanish)
transformer model, I wondered how many parameters a neural network
needs to "store" each individual vocabulary item.
This grossly oversimplified experiment is the result.

## The experiment

I create a synthetic bilingual vocabulary of randomly generated
words consisting of 5-10 English letters.
Mimicking a natural-language bilingual vocabulary list,
each vocabulary item is a pair, representing a word in the input language
and its "translation" in the output language.

I then train a sequence-to-sequence model on this vocabulary
(using character-level encoding).
The model should learn, given as input the first word in some vocabulary pair,
to output the second.

I train and test models of various scales on vocabularies of various sizes 
(16, 64, 256, or 1024 pairs of words),
and determine the smallest number of parameters needed to learn
all the words in the vocabulary.


## Results

I expected the number of parameters to grow
linearly in the vocabulary size, or maybe slightly faster.
The model needs to somehow "store" the full vocabulary list in its weights,
so on information-theoretic grounds, the number of parameters must grow
at least linearly in the vocabulary.
At the same time, the "translation" task becomes more complex
when the number of vocabulary items is large compared to the number of tokens,
since the model needs to learn to distinguish between items that share 
common tokens.
This requires additional complexity, maybe causing superlinear growth 
in the number of parameters.

Here are results from two runs of the experiment.
(The training process is randomized; your results may be slightly different.)

| Vocabulary size | Num parameters needed to learn all items (First run, second run) | Parameters per item |
| -- | -- | -- |
| 16 | 2053, 2053 | 128, 128 |
| 64 | 8521, 12913 | 133, 202 |
| 256 | 70081, 70081 | 274, 274 |
| 1024 | N/A | N/A |

With 1024 words, even the largest model (189201 params) did not achieve perfect test performance.

The results are noisy -- they vary from experiment to experiment -- 
but overall consistent with linear or slightly superlinear growth
in model size as a function of vocabulary size.

## Running the experiment

The vocabulary has already been generated; it can be found in `data/vocab.txt`.
The code used to generate it is in `gen_vocab.py`.

To run the experiment you will need to have Pytorch installed.

To run the experiment, simply `cd src` and then run `python controller.py > ../logs/output`.
Progress will be written to the terminal.
Detailed results per experiment will be written to log files in `logs/`, and
when the experiment concludes, a summary will be written to `logs/output`.

The full experiment runs 160 experiments and takes several hours to run.
You have several options to speed it up:
- Run fewer experiments.  Change the arrays of parameter options in `controller.py:experiment()`.
- Run fewer training cycles on each experiment.  Change the `epochs` value in `config.py`.  For the smallest vocabulary size of 16, it may take up to 2000 training cycles to see results.

## Limitations

Unlike most machine learning models, this model is intended to overfit to its training data:
the model is not tested on any data other than the specific items on which it was trained.
A true machine translation model would have to generalize from its training data
but still (implicitly) learn translations of every word in the language;
I expect this to require far more parameters.

It would be interesting to explore dependency on the number of tokens as well,
since this is a configurable parameter in ML applications.
