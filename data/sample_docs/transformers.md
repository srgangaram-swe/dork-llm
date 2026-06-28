# Transformers and Self-Attention

The transformer is a neural network architecture built around the attention
mechanism. A decoder-only transformer, such as the GPT family, predicts the next
token given all previous tokens.

## Self-attention

Self-attention lets each token attend to every other token in the sequence. Each
token is projected into a query, a key, and a value vector. Attention weights are
computed as the scaled dot product of queries and keys, passed through a softmax,
and used to take a weighted sum of the values. Multi-head attention runs several
attention operations in parallel and concatenates their outputs.

## Causal masking

In a decoder, causal masking prevents a position from attending to future
positions. This preserves the autoregressive property: the prediction for token t
depends only on tokens before t. Without causal masking the model could trivially
cheat by looking at the answer.

## Positional information

Attention is permutation invariant, so transformers add positional information.
Learned positional embeddings store a vector per position. Sinusoidal embeddings
use fixed sine and cosine functions. Rotary positional embeddings (RoPE) rotate
the query and key vectors by an angle proportional to their position.

## Residual stream and layer normalization

Each transformer block applies attention and a feedforward network with residual
connections. Pre-norm architectures apply layer normalization before each
sub-layer, which stabilizes training of deep networks.
