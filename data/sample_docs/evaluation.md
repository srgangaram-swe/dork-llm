# Evaluating Language Models

Evaluation matters because fluent text is not the same as correct or safe text.
Before deploying an LLM feature, teams run an evaluation harness that scores the
model across many axes and gates releases on the results.

## Perplexity

Perplexity is the exponential of the average negative log-likelihood the model
assigns to held-out text. Lower perplexity means the model predicts the corpus
better. Perplexity measures language-modeling quality but not task correctness.

## Task metrics

Exact match and token F1 score short-answer accuracy. Multiple-choice accuracy
checks whether the model selects the correct option. JSON validity measures
whether structured outputs parse and contain the required keys.

## Faithfulness and tool use

For RAG, faithfulness checks whether answers are grounded in the cited context,
and citation coverage checks whether citations are present and valid. For agents,
tool-use accuracy checks whether the model selects the correct tool and arguments.

## Performance and regression testing

Latency and throughput measure serving cost. Regression tests pin expected
behavior so that a change which degrades quality fails continuous integration.
