# Training a Small Language Model

Training a decoder-only transformer from scratch follows a standard recipe:
prepare a corpus, train a tokenizer, and optimize the model with next-token
prediction.

## Tokenization

A tokenizer maps text to integer ids. Byte-level BPE merges frequent byte pairs
into subword tokens and never emits an unknown token, because its base alphabet
is the 256 bytes. A character-level tokenizer maps each character to an id and
needs no training.

## Optimization

The model is trained with the cross-entropy loss between predicted and actual
next tokens. AdamW is the standard optimizer, with weight decay applied to weight
matrices but not to biases or layer-norm parameters. A cosine learning-rate
schedule with linear warmup is common.

## Checkpointing and validation

Checkpointing saves the model weights, optimizer state, and configuration so a
run can resume or be reloaded for inference. Periodic validation on held-out data
detects overfitting; the best checkpoint by validation loss is kept.

## Sampling

At generation time the model produces a probability distribution over the next
token. Temperature rescales the logits, top-k keeps the k most likely tokens, and
top-p (nucleus) sampling keeps the smallest set of tokens whose cumulative
probability reaches p.
