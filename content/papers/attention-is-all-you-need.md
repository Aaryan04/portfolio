---
title: Attention Is All You Need
type: paper
subtitle: The Transformer, explained for a first-time reader
authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
affiliation: Google Brain, Google Research and the University of Toronto. All eight are listed as equal contributors, in random order.
venue: NeurIPS 2017
year: 2017
paper_url: https://arxiv.org/abs/1706.03762
paper_label: arXiv:1706.03762
date: 2026-10-12
summary: The paper that replaced recurrence with self-attention and became the basis of BERT, GPT and nearly every modern language model. Every section, table and ablation, with live demos of attention, scaling, positional encoding and the learning-rate schedule.
tags: Transformers, Attention, NLP, Deep learning
prerequisites: A rough idea of what a neural network is. Everything else is explained as we go.
featured: true
---

## TL;DR

- In 2017, the best translation systems read a sentence **one word at a time** using recurrent neural networks (RNNs). That made them slow to train, because step 50 had to wait for step 49.
- The paper proposes the **Transformer**, which drops recurrence completely. Every word looks at every other word at once through **attention**, so a whole sentence is processed in parallel.
- The core operation is **scaled dot-product attention**, $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$, run as several **heads** in parallel. **Positional encodings** tell the model about word order, since attention by itself ignores order.
- On English-to-German translation (WMT 2014) it scored **28.4 BLEU**, more than 2 points above the best previous systems including ensembles. On English-to-French it set a single-model record of **41.8 BLEU**. Training took **3.5 days on 8 GPUs**, a fraction of previous costs.
- It also did well on English **constituency parsing**, a very different task, with little tuning. This architecture later became the basis of BERT, GPT and almost every large language model.

## How this article is organised

It follows the paper's own order, adding background where a first-time reader needs it. Section numbers in brackets, like (§3.2), point to the paper.

1. The problem and what came before (§1–2)
2. The architecture, piece by piece (§3)
3. Why self-attention is a good idea (§4)
4. How it was trained (§5)
5. Results, ablations and parsing (§6)
6. Attention visualisations (appendix)
7. The authors' conclusion (§7), then a short "reading it today" note, a glossary and a cheat sheet

---

## 1. The problem: sequence transduction in 2017 (§1)

**Sequence transduction** means turning one sequence into another: a sentence in English into one in German, speech into text, a sentence into its grammatical parse. The paper's main test is **machine translation**.

### Encoder-decoder models

The standard design was an **encoder-decoder**:

- The **encoder** reads the input sentence and turns it into vectors (lists of numbers) that represent its meaning.
- The **decoder** produces the output one token at a time. Each new token is predicted from the encoder's vectors and from the tokens already produced. Feeding a model's own previous outputs back in like this is called being **auto-regressive**.

Both halves were usually **recurrent neural networks**, mostly LSTMs or GRUs. An RNN keeps a hidden state $h_t$ and updates it once per word: $h_t = f(h_{t-1}, x_t)$. To compute the state for word 50, you need the state for word 49, and so on back to word 1.

That sequential dependency is the problem the paper attacks. GPUs are fast because they do thousands of things in parallel, but an RNN must take its steps in order **within** each sentence. The paper notes that this "precludes parallelization within training examples, which becomes critical at longer sequence lengths", since memory limits how many sentences you can batch together. Tricks such as factorised layers and conditional computation sped RNNs up, but "the fundamental constraint of sequential computation, however, remains."

### Attention, before the Transformer

**Attention** already existed. Introduced for translation by Bahdanau, Cho and Bengio in 2014, it lets the decoder, at each output step, look back over *all* encoder states and take a weighted average of them, with weights saying how relevant each input word is right now. When producing the German verb, the decoder can focus on the English verb even if it was 20 words earlier.

Attention lets a model connect words "without regard to their distance". But until this paper it was almost always added *on top of* an RNN. The paper's question is: what if attention is the only mechanism?

> **The claim in one line:** "We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output." It reached a new state of the art in translation "after being trained for as little as twelve hours on eight P100 GPUs."

## 2. What came before (§2)

Others had also tried to remove the sequential bottleneck, mainly with **convolutional neural networks** (CNNs), which process all positions in parallel: the Extended Neural GPU, **ByteNet** and **ConvS2S**. A convolution only looks at a small window of neighbours, though. To connect two distant words you must stack layers. The number of operations needed to relate two positions grows **linearly** with their distance in ConvS2S and **logarithmically** in ByteNet. That makes long-range dependencies harder to learn.

In the Transformer, any two positions are connected in a **constant** number of operations. The paper admits a cost: an attention output is a weighted *average*, which blurs detail ("reduced effective resolution"). Its fix is **multi-head attention**, covered below.

The paper also credits earlier uses of **self-attention**, meaning attention between positions of the *same* sequence. It had been used for reading comprehension, abstractive summarisation, textual entailment and sentence representations. It also cites end-to-end memory networks, which use recurrent attention for question answering. The novelty claimed is that the Transformer is "the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."

---

## 3. The architecture (§3)

The Transformer keeps the encoder-decoder shape. The encoder maps input tokens $(x_1, \dots, x_n)$ to vectors $\mathbf{z} = (z_1, \dots, z_n)$. Given $\mathbf{z}$, the decoder generates the output $(y_1, \dots, y_m)$ one token at a time, auto-regressively.

```demo architecture
The Transformer, after the paper's Figure 1. Click any block to see what it does and where the paper describes it.
```

### 3.1 The encoder and decoder stacks (§3.1)

**Encoder.** A stack of $N = 6$ identical layers. Each layer has two **sub-layers**:

1. multi-head **self-attention**, where every input position looks at every input position;
2. a simple position-wise **feed-forward network**, applied to each position separately.

Each sub-layer is wrapped in a **residual connection** followed by **layer normalisation**:

$$\text{output} = \mathrm{LayerNorm}\big(x + \mathrm{Sublayer}(x)\big)$$

- A **residual connection** adds the sub-layer's input back to its output. The layer only has to learn a *change* to its input, which makes deep stacks much easier to train. This idea came from ResNets in image recognition.
- **Layer normalisation** rescales each position's vector to zero mean and unit variance (with learned scale and shift), which keeps values in a stable range.

To make the residual additions work, every sub-layer and the embedding layers output vectors of the same size, $d_{\text{model}} = 512$.

**Decoder.** Also $N = 6$ identical layers, with a third sub-layer in between: multi-head attention over the **encoder's output**. This is where the decoder "reads" the source sentence. The decoder's self-attention is **masked** so that position $i$ cannot see positions after $i$. Because the output is also shifted right by one position, the prediction for position $i$ depends only on outputs before $i$. During training, the model sees the whole target sentence at once, but each position must predict its next token without peeking ahead.

### 3.2 Attention (§3.2)

The paper defines attention as "mapping a query and a set of key-value pairs to an output". The output is a weighted sum of the **values**, and each weight comes from how well the **query** matches the corresponding **key**.

> **Analogy:** A library catalogue. Your question is the **query**. Each book has a catalogue card, its **key**, and the book itself, its **value**. Attention compares your query with every card, then hands you a blend of all the books, mostly made of those whose cards matched best. Unlike a real library, the match is soft: you get 58% of one book, 28% of another and 14% of a third.

In self-attention, the queries, keys and values are all computed from the same sentence. Each word asks "which other words matter for understanding me?", and its new representation is a blend of the words it attends to.

#### Scaled dot-product attention (§3.2.1)

Stack the queries into a matrix $Q$, the keys into $K$ and the values into $V$. Queries and keys have dimension $d_k$; values have dimension $d_v$.

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Reading it from the inside out:

1. $QK^\top$: the dot product of every query with every key, giving a matrix of match scores. For $n$ words, it is $n \times n$.
2. $\div \sqrt{d_k}$: scale the scores down (explained just below).
3. **softmax**, row by row: turn each row of scores into positive weights that sum to 1.
4. $\times V$: each output is the weighted average of the value vectors.

**A tiny worked example.** Take the sentence "the animal … street … it" and focus on the query for "it", with $d_k = 2$. Suppose:

| word | key | value |
|---|---|---|
| the | (0, 1) | (1, 0) |
| animal | (2, 0) | (0, 1) |
| street | (1, 1) | (1, 1) |

With query $q_{\text{it}} = (1, 0)$:

- dot products with the keys are 0, 2 and 1;
- divided by $\sqrt{2}$, they become 0, 1.41 and 0.71;
- softmax turns those into weights of **0.14, 0.58 and 0.28**, so "it" attends mostly to "animal";
- the output is $0.14\,(1,0) + 0.58\,(0,1) + 0.28\,(1,1) = (0.42,\ 0.86)$, a new vector for "it" that now carries mostly "animal" information.

In a real model, queries, keys and values are learned, so the network discovers what "matching" should mean.

```demo attention
The worked example, live. Rotate the query for "it" and watch the attention weights and the output vector change. Toggle the √d_k scaling to see how it softens the weights.
```

**Why scale by $\sqrt{d_k}$?** The two common attention types were *additive* attention, which scores with a small one-layer network, and *dot-product* attention. Dot-product attention is much faster and uses less memory because it is just matrix multiplication, which GPUs are built for. But for large $d_k$, unscaled dot-product attention did worse than additive attention. The authors' explanation: if the components of $q$ and $k$ are independent with mean 0 and variance 1, their dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and **variance $d_k$**. With $d_k = 64$, scores have a standard deviation of about 8. Softmax over numbers that spread out becomes nearly one-hot: one weight close to 1, the rest close to 0. In that region its gradients are tiny, so learning stalls. Dividing by $\sqrt{d_k}$ brings the variance back to 1.

```demo scaling
Why the paper scales by √d_k. Eight random keys are scored against a random query with d_k dimensions. Raise d_k: without scaling, softmax collapses onto one key; with scaling, the weights stay spread out.
```

**Masking.** In the decoder, a position must not attend to future positions. The fix is to set the scores for illegal connections to $-\infty$ before the softmax. Since $e^{-\infty} = 0$, those positions get exactly zero weight.

#### Multi-head attention (§3.2.2)

One attention operation produces one weighted average per position. But a word may need several kinds of information at once: its subject, its object, the word before it. Averaging them all into one blend loses detail. The paper's answer is to run several attentions in parallel, each with its own learned projections:

$$\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^O$$
$$\mathrm{head}_i = \mathrm{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

- Each head projects the $d_{\text{model}}$-dimensional input down to $d_k$ dimensions for queries and keys and $d_v$ for values. The projection matrices are $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.
- The $h$ head outputs are concatenated and projected back to $d_{\text{model}}$ by $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.
- The paper uses $h = 8$ heads with $d_k = d_v = d_{\text{model}}/h = 64$. Because each head is smaller, total compute is similar to one big head with full dimensionality.

In the paper's words: "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."

> **Analogy:** One reader highlighting a page with one colour, versus eight readers each highlighting for one thing (names, verbs, dates, pronouns), then combining their notes.

#### Three ways the Transformer uses attention (§3.2.3)

1. **Encoder self-attention.** Queries, keys and values all come from the previous encoder layer. Every input position can attend to every input position.
2. **Decoder (masked) self-attention.** Each output position attends to itself and earlier output positions only, which preserves the auto-regressive property.
3. **Encoder-decoder attention.** Queries come from the decoder; keys and values come from the encoder's final output. Every decoder position can look at the whole source sentence, like the RNN attention described in section 1.

### 3.3 Position-wise feed-forward networks (§3.3)

After attention mixes information *between* positions, each position is processed *on its own* by the same small two-layer network:

$$\mathrm{FFN}(x) = \max(0,\ x W_1 + b_1)\, W_2 + b_2$$

$\max(0, \cdot)$ is the ReLU activation. The inner layer is wider, $d_{ff} = 2048$, with input and output $d_{\text{model}} = 512$. The same weights are used at every position within a layer, but each layer has its own. The paper notes this is equivalent to two convolutions with kernel size 1.

### 3.4 Embeddings and softmax (§3.4)

- **Embeddings:** each input and output token is turned into a learned vector of size $d_{\text{model}}$.
- **Output:** the decoder's final vectors go through a learned linear layer and a **softmax**, which gives a probability for every token in the vocabulary as the next token.
- **Weight sharing:** the same matrix is used for the input embedding, the output embedding and the pre-softmax linear layer, which saves parameters.
- **Scaling:** in the embedding layers, the weights are multiplied by $\sqrt{d_{\text{model}}}$.

### 3.5 Positional encoding (§3.5)

Attention compares every word with every other word but has **no idea of order**. Shuffle the input words and each word's set of attention scores stays the same. Since "dog bites man" and "man bites dog" mean different things, the model needs position information. The Transformer adds a **positional encoding** vector to each token's embedding at the bottom of both stacks. The paper uses fixed sine and cosine waves of different frequencies:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$pos$ is the position in the sentence and $i$ indexes the dimension. Each pair of dimensions is a sine and cosine pair at one frequency. The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

> **Analogy:** A clock with many hands. The second hand spins fast and tells you fine position; the hour hand moves slowly and tells you coarse position. Together, the hands give every moment a unique reading. Each sine/cosine pair is one hand.

**Why sinusoids?** The authors hypothesised that this makes it easy to attend by *relative* position. For any fixed offset $k$, $PE_{pos+k}$ is a **linear function** of $PE_{pos}$: for each frequency $\omega$, shifting by $k$ is just a rotation of the $(\sin \omega\, pos, \cos \omega\, pos)$ pair by angle $\omega k$. They also tried **learned** positional embeddings, one learned vector per position, and got nearly identical results (Table 3, row E). They kept sinusoids because they "may allow the model to extrapolate to sequence lengths longer than the ones encountered during training."

```demo positional
The sinusoidal positional encodings for the first 64 positions of a 128-dimensional model. Each row is one position; early dimensions oscillate quickly, later ones slowly. Hover or tap a row to see its values.
```

---

## 4. Why self-attention? (§4)

The paper compares self-attention layers with recurrent and convolutional layers on three criteria:

1. **Computation per layer:** how much work one layer takes.
2. **Sequential operations:** how many steps must happen one after another, which limits parallelism.
3. **Maximum path length:** how many steps a signal must travel between the two most distant positions. Shorter paths make long-range dependencies easier to learn.

Here $n$ is the sequence length, $d$ the representation dimension, $k$ the convolution kernel size and $r$ the neighbourhood size for restricted attention.

| Layer type | Complexity per layer | Sequential operations | Maximum path length |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention (restricted) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

*Table 1 of the paper.*

How to read it:

- **Self-attention connects everything in one step:** path length $O(1)$, and all positions are computed in parallel.
- **Cost:** self-attention is quadratic in $n$, recurrence quadratic in $d$. Self-attention is cheaper whenever $n < d$. That is the usual case for sentences in translation: a few dozen subword tokens versus $d = 512$.
- **Very long sequences:** attention could be **restricted** to a window of $r$ neighbours, at the price of longer paths. The paper leaves this for future work.
- **Convolutions:** one layer with $k < n$ does not connect all pairs. You need a stack of $O(n/k)$ ordinary layers or $O(\log_k n)$ dilated ones, and convolutions are generally more expensive than recurrent layers by a factor of $k$. Separable convolutions cut the cost to $O(k \cdot n \cdot d + n \cdot d^2)$, but even with $k = n$, that equals the cost of a self-attention layer plus a feed-forward layer, which is exactly what the Transformer uses.

**A bonus: interpretability.** You can look at attention weights to see what the model attends to. The authors found that individual heads learn different tasks, many of which "appear to exhibit behavior related to the syntactic and semantic structure of the sentences" (see section 9 below).

---

## 5. Training (§5)

### Data and batching (§5.1)

- **English-German:** the WMT 2014 dataset, about **4.5 million sentence pairs**. Text was split into subword units with **byte-pair encoding (BPE)**, using a shared source and target vocabulary of about **37,000 tokens**. BPE repeatedly merges the most frequent pairs of characters or pieces, so common words stay whole and rare words become several pieces.
- **English-French:** the much larger WMT 2014 set, **36 million sentences**, with a **32,000** word-piece vocabulary.
- **Batching:** sentence pairs were grouped by similar length. Each training batch held roughly **25,000 source tokens and 25,000 target tokens**.

### Hardware and schedule (§5.2)

One machine with **8 NVIDIA P100 GPUs**.

| Model | Time per step | Steps | Wall-clock time |
|---|---|---|---|
| Base | ~0.4 s | 100,000 | ~12 hours |
| Big | ~1.0 s | 300,000 | 3.5 days |

### Optimiser and learning-rate schedule (§5.3)

The optimiser was **Adam** with $\beta_1 = 0.9$, $\beta_2 = 0.98$ and $\epsilon = 10^{-9}$. The learning rate follows a now-famous schedule:

$$\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\big(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5}\big), \qquad \text{warmup\_steps} = 4000$$

- During the first 4,000 steps, the rate rises **linearly** (warm-up).
- After that, it decays in proportion to $1/\sqrt{\text{step}}$.
- With $d_{\text{model}} = 512$, it peaks at about $7 \times 10^{-4}$ at step 4,000 and falls to about $1.4 \times 10^{-4}$ by step 100,000.
- Warm-up helps because early updates, made while the model is still random, can be large and destabilising.

```demo schedule
The paper's learning-rate schedule. Change the warm-up length and model width to see how the peak and the decay move.
```

### Regularisation (§5.4)

- **Residual dropout.** Dropout randomly zeroes a fraction of activations during training so the network cannot rely on any single path. It is applied to the output of each sub-layer before the residual addition and normalisation, and to the sum of embeddings and positional encodings. Rate $P_{\text{drop}} = 0.1$ for the base model.
- **Label smoothing** with $\epsilon_{ls} = 0.1$. Instead of training toward 100% probability on the correct token, the target puts 90% on it and spreads 10% over the rest. In the paper's words, this "hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."

---

## 6. Results (§6)

### 6.1 Machine translation (§6.1)

**BLEU** is the standard automatic translation score. It measures how many word sequences of length 1 to 4 in the system's output also appear in human reference translations, with a penalty for outputs that are too short. Higher is better; gains of 1 to 2 points were considered significant at the time.

| Model | EN→DE BLEU | EN→FR BLEU | Training cost (FLOPs) |
|---|---|---|---|
| ByteNet | 23.75 | | |
| GNMT + RL | 24.6 | 39.92 | |
| ConvS2S | 25.16 | 40.46 | |
| MoE | 26.03 | 40.56 | |
| GNMT + RL (ensemble) | 26.30 | 41.16 | |
| ConvS2S (ensemble) | 26.36 | 41.29 | |
| **Transformer (base)** | **27.3** | **38.1** | $3.3 \times 10^{18}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \times 10^{19}$ |

*Selected rows of Table 2, tested on newstest2014. An ensemble combines several separately trained models, which usually adds accuracy at a large cost.*

- **English-German:** the big Transformer beats the best previously reported models, **including ensembles, by more than 2.0 BLEU**. Even the base model surpasses all previous models and ensembles, at a fraction of their training cost.
- **English-French:** the big model reaches **41.8**, a new single-model state of the art, at less than a quarter of the previous best model's training cost. The current arXiv version reports 41.8; some earlier printings, including the conference proceedings, report 41.0. The English-French big model used $P_{\text{drop}} = 0.1$ instead of 0.3.
- **Training cost:** estimated as training time × number of GPUs × an estimate of each GPU's sustained single-precision throughput. The paper uses 2.8, 3.7, 6.0 and 9.5 TFLOPS for the K80, K40, M40 and P100.

Decoding details matter for these numbers:

- **Checkpoint averaging:** the base models use the average of the last 5 checkpoints, saved 10 minutes apart; the big models average the last 20.
- **Beam search** with beam size 4: instead of greedily taking the most likely token at each step, keep the 4 best partial translations.
- **Length penalty** $\alpha = 0.6$, so long translations are not unfairly penalised.
- **Output length:** at most the input length + 50 tokens, stopping early when possible.

The beam size and length penalty were chosen on the development set.

### 6.2 What matters: model variations (§6.2)

To see which components matter, the authors changed the base model one piece at a time. They measured perplexity (PPL, lower is better) and BLEU on the English-German development set (newstest2013), without checkpoint averaging. The perplexities are per word-piece, so they should not be compared with per-word perplexities.

| Variation | PPL | BLEU | Params |
|---|---|---|---|
| **Base** ($N{=}6$, $d_{\text{model}}{=}512$, $d_{ff}{=}2048$, $h{=}8$, $d_k{=}d_v{=}64$, dropout 0.1, $\epsilon_{ls}{=}0.1$, 100K steps) | 4.92 | 25.8 | 65M |
| (A) 1 head, $d_k{=}512$ | 5.29 | 24.9 | |
| (A) 32 heads, $d_k{=}16$ | 5.01 | 25.4 | |
| (B) $d_k{=}16$ (8 heads) | 5.16 | 25.1 | 58M |
| (C) $d_{\text{model}}{=}1024$ | 4.66 | 26.0 | 168M |
| (C) $d_{ff}{=}4096$ | 4.75 | 26.2 | 90M |
| (D) no dropout | 5.77 | 24.6 | |
| (D) no label smoothing | 4.67 | 25.3 | |
| (E) learned positional embeddings | 4.92 | 25.7 | |
| **Big** ($d_{\text{model}}{=}1024$, $d_{ff}{=}4096$, $h{=}16$, dropout 0.3, 300K steps) | 4.33 | 26.4 | 213M |

*Selected rows of Table 3.*

What the authors conclude:

- **(A) Number of heads.** With compute held constant, a single head is **0.9 BLEU worse** than the best setting, and quality also drops with too many heads.
- **(B) Key size.** Shrinking $d_k$ hurts, suggesting that "determining compatibility is not easy" and that a more sophisticated compatibility function than the dot product might help.
- **(C) Size.** Bigger models are better.
- **(D) Regularisation.** Dropout is very helpful against overfitting. Label smoothing raises perplexity but improves BLEU: without it, perplexity improves to 4.67 while BLEU drops to 25.3.
- **(E) Positions.** Learned positional embeddings give nearly identical results to sinusoids.

### 6.3 Beyond translation: English constituency parsing (§6.3)

To test whether the Transformer generalises, the authors tried **constituency parsing**: turning a sentence into a tree of nested phrases, written out as a bracketed sequence such as `(S (NP The cat) (VP sat))`. This task is hard for sequence models because the output is strongly constrained by structure and much longer than the input. RNN sequence-to-sequence models had not reached the state of the art when training data was small.

Setup:

- A **4-layer** Transformer with $d_{\text{model}} = 1024$.
- **WSJ only:** trained on the Wall Street Journal portion of the Penn Treebank, about **40,000** training sentences, with a 16K-token vocabulary.
- **Semi-supervised:** also trained on larger high-confidence and BerkeleyParser corpora, about **17 million** sentences, with a 32K-token vocabulary.
- Only dropout, learning rates and beam size were tuned, on the development set (Section 22). Everything else was unchanged from the English-German base model.
- Decoding used a maximum output length of input length + 300, beam size 21 and $\alpha = 0.3$.

Results on WSJ Section 23, as F1 scores (higher is better):

| Training data | Transformer F1 | Notes |
|---|---|---|
| WSJ only (40K sentences) | **91.3** | Beats the BerkeleyParser (90.4), which RNN seq2seq models had failed to do on this little data |
| Semi-supervised | **92.7** | |

The Transformer did better than all previously reported models except the Recurrent Neural Network Grammar, despite having almost no task-specific tuning.

---

## 7. Peeking inside: attention visualisations (appendix)

The appendix shows attention weights from encoder self-attention in layer 5 of 6 for a sentence about American governments passing laws "making the registration or voting process more difficult":

- **Long-distance dependencies (Figure 3).** Many heads attending from "making" focus on the distant word "difficult", completing the phrase "making … more difficult".
- **Anaphora resolution (Figure 4).** Two heads appear to work out what the pronoun "its" refers to, with very sharp attention from "its".
- **Sentence structure (Figure 5).** Different heads clearly learn different, structure-related behaviours.

These plots are illustrations, not proof of what the model computes, but they made attention a popular tool for interpretation.

## 8. The authors' conclusion (§7)

The authors summarise the Transformer as "the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention". For translation, it "can be trained significantly faster than architectures based on recurrent or convolutional layers."

Their stated future work:

- apply attention-based models to other tasks, including input and output modalities other than text;
- investigate **local, restricted attention** to handle large inputs and outputs such as **images, audio and video**;
- make generation **less sequential**.

The code was released in Google's **tensor2tensor** library. The paper's contribution footnote credits:

| Author | Contribution |
|---|---|
| Jakob Uszkoreit | Proposed replacing RNNs with self-attention |
| Ashish Vaswani and Illia Polosukhin | Designed and implemented the first Transformer models |
| Noam Shazeer | Proposed scaled dot-product attention, multi-head attention and the parameter-free position representation |
| Niki Parmar | Designed, implemented, tuned and evaluated countless model variants |
| Llion Jones | Worked on model variants, the initial codebase, efficient inference and visualisations |
| Łukasz Kaiser and Aidan Gomez | Designed and implemented tensor2tensor |

---

## 9. Reading it today

*This section is context from after 2017, not from the paper.*

- **The future work happened, and then some.** Within a year and a half, BERT (encoder only) and GPT (decoder only) showed the architecture could be pre-trained on huge amounts of text. The Vision Transformer (2020) applied it to images, and similar models now handle audio, video, code and proteins. Almost every large language model is a decoder-only Transformer.
- **Some details changed.** Most modern models normalise *before* each sub-layer ("pre-LN"), which trains more stably in deep stacks. Sinusoidal positions were largely replaced by learned or rotary schemes, and ReLU often became GELU or gated variants.
- **The quadratic cost became the main bottleneck.** The $O(n^2)$ attention cost in Table 1 barely mattered for 50-token sentences. It became the central engineering problem once models read thousands or millions of tokens, which led to work on efficient attention (the paper's "restricted attention" idea), FlashAttention and KV caching.
- **The core is unchanged.** Scaled dot-product attention, multiple heads, residual connections, normalisation and position-wise feed-forward layers are still the backbone.

## 10. Glossary

| Term | Meaning |
|---|---|
| Auto-regressive | Generating one token at a time, each conditioned on the previous ones. |
| Self-attention | Attention where queries, keys and values all come from the same sequence. |
| Query, key, value | What a position is looking for, what it offers to be matched on, and what it contributes if matched. |
| Head | One of several parallel attention operations with their own projections. |
| Masking | Blocking attention to certain positions (here, future tokens) by setting their scores to $-\infty$. |
| Residual connection | Adding a layer's input to its output, so the layer learns a change. |
| Layer normalisation | Rescaling each vector to a standard range for stable training. |
| Positional encoding | A vector added to embeddings so the model knows token order. |
| BPE / word-piece | Ways of splitting text into subword units. |
| BLEU | An automatic translation score based on overlapping word sequences with reference translations. |
| Perplexity | How surprised a model is by the correct text; lower is better. |
| Beam search | Decoding that keeps the top-$k$ partial outputs at each step. |
| Label smoothing | Training toward slightly-less-than-certain targets, which improves generalisation. |
| F1 | The harmonic mean of precision and recall, used for parsing accuracy. |

## 11. Cheat sheet: the numbers to remember

| | Base | Big |
|---|---|---|
| Layers ($N$) | 6 encoder + 6 decoder | 6 + 6 |
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| Heads ($h$) | 8 | 16 |
| $d_k = d_v$ | 64 | 64 |
| Dropout | 0.1 | 0.3 (0.1 for EN→FR) |
| Parameters | 65M | 213M |
| Training | 100K steps, ~12 h on 8 P100s | 300K steps, 3.5 days on 8 P100s |
| EN→DE BLEU | 27.3 | 28.4 |
| EN→FR BLEU | 38.1 | 41.8 |

**Shared settings:**
- Adam ($\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$) with 4,000 warm-up steps
- Label smoothing 0.1
- Beam size 4, length penalty 0.6
- Batches of about 25K source + 25K target tokens

**The equation to remember:** $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$

---

*Numbers are taken from the paper's tables and text; section 9 is the only part that goes beyond the paper. Interactive figures use illustrative values.*
