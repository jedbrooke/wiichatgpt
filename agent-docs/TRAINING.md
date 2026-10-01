# WiiChatGPT Training Plan

## Training Hardware

RTX 5080, 16 GB VRAM. Memory budget for the 288M parameter model (see MODEL.md):

| Component | Size |
|---|---|
| Parameters (288M x 4 bytes, fp32) | ~1.15 GB |
| Gradients (288M x 4 bytes) | ~1.15 GB |
| AdamW optimizer states (m + v, 288M x 4 x 2) | ~2.3 GB |
| **Fixed overhead** | **~4.6 GB** |

Activation memory per sample (seq_len=512): ~200 MB (attention scores + FFN intermediates across 20 layers).

| Batch size | Activations | Total VRAM |
|---|---|---|
| 8 | ~1.6 GB | ~6.2 GB |
| 16 | ~3.2 GB | ~7.8 GB |
| 32 | ~6.4 GB | ~11.0 GB |
| 48 | ~9.6 GB | ~14.2 GB |

Batch sizes of 32-48 fit without gradient checkpointing, model parallelism, or offloading. Mixed precision (bf16 on 5080 tensor cores) roughly halves parameter and gradient memory, leaving room for even larger batches or longer context.

Estimated training throughput: 50-100K tokens/second at bf16.

## Data

### Domain Corpus

Target domain: Wii trivia, popular Wii games, Nintendo lore.

| Source | Estimated tokens |
|---|---|
| Wikipedia (Wii games, console, accessories, developers) | 20-40M |
| Game guides, walkthroughs, reviews (GameFAQs, IGN, etc.) | 50-100M |
| Forum/community content (Reddit r/wii, r/nintendo, etc.) | 50-200M |
| **Total domain** | **100-300M** |

### General English Supplement

A subset of clean general English text (OpenWebText, WikiText, or similar open datasets) mixed at 15-20% of the total corpus. Purpose: teach grammar, sentence structure, and discourse flow that may be underrepresented in domain-only content.

### Synthetic Data

Use a larger model (e.g., Qwen3 0.6B or similar) to generate supplementary training data:
- Q&A pairs about Wii games and features
- Trivia questions with explanations
- Conversational dialogues about Wii topics
- Game descriptions and summaries
- Rephrased/simplified versions of existing content

This can 2-3x the effective corpus size.

### Corpus Targets

| Variant | Source tokens | x Epochs | Effective tokens seen |
|---|---|---|---|
| Domain only | 100-300M | 3-4 | 300M-1.2B |
| + general English (15%) | 120-350M | 3-4 | 360M-1.4B |
| + synthetic data | 200-600M | 3-4 | 600M-2.4B |

Chinchilla-optimal for 288M params would be ~5.8B tokens. The domain is too narrow to source that much high-quality data. Multi-epoch training and synthetic augmentation close the gap. Small models tolerate repetition well, with continued improvement through 4+ epochs empirically.

At 50-100K tokens/sec on the 5080, even 2.4B effective tokens is only 7-13 hours of training.

## Tokenizer

32K BPE vocabulary trained with SentencePiece (as specified in MODEL.md). Trained on the combined corpus (domain + general English) before model training begins.

The tokenizer model (~2 MB) is exported alongside the trained weights for Wii deployment.

## Training Approaches

Four approaches in order of complexity. Start with Approach 1. Escalate only if evaluation shows the model's English fluency or domain accuracy is insufficient.

All approaches share the same data pipeline, tokenizer, evaluation harness, and export process. The only difference is how the training loop is configured.

### Approach 1: Domain-Only Training from Scratch

Single training run on domain data only. The domain corpus (Wikipedia, game guides, forum posts) is already well-formed English, so a separate general English supplement may not be necessary.

**Suggested Training configuration:** (will require experimentation)
- Optimizer: AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- Learning rate: 3e-4 peak, cosine decay to 3e-5
- Warmup: 1-2% of total steps
- Batch size: 32-48 (depending on VRAM headroom after bf16)
- Sequence length: 512 (matching inference context)
- Epochs: 3-4 over the domain corpus (+ synthetic data if available)
- Precision: bf16 mixed precision (fp32 master weights)

**All parameters trained jointly:** dense core, embeddings, and PLE table receive gradients via standard backpropagation. The PLE table is a large embedding lookup; its gradients are sparse (only rows corresponding to tokens in the current batch are updated).

**Expected outcome:**
- Grammar and sentence structure: likely functional (domain sources are English wiki/forum text)
- Domain knowledge: solid
- Off-topic ability: essentially zero (acceptable for a Wii chatbot)

### Approach 1.5: Blended Training from Scratch

If Approach 1 produces occasional grammar or fluency issues, add a general English supplement.

**Corpus blend:** ~80% domain content, ~20% general English (OpenWebText, WikiText, or similar). Shuffled together at the document level. Everything else identical to Approach 1.

The general English fraction provides broader sentence structures and discourse patterns that may be underrepresented in domain-only content.

### Approach 2: Two-Phase Training from Scratch

If Approach 1 produces weak English fluency despite good domain knowledge.

**Phase 1 (general English pretraining):**
- Train the full PLE architecture on 2-3B tokens of general English (OpenWebText or similar)
- Same optimizer and schedule as Approach 1
- Goal: the model learns grammar, syntax, and coherent sentence formation
- Produces an evaluable checkpoint (useful debugging milestone)

**Phase 2 (domain fine-tuning):**
- Continue training from the Phase 1 checkpoint on domain corpus
- Lower learning rate (1e-4 peak, cosine decay to 1e-5)
- 3-4 epochs over domain data
- Optionally retain 5-10% general English in the mix to reduce catastrophic forgetting

**What changes vs Approach 1:** two separate training runs with different data and learning rates. Hyperparameter tuning for the phase transition (when to stop phase 1, how aggressively to fine-tune in phase 2).

**What stays the same:** data pipeline, tokenizer, model architecture, export process.

### Approach 3: Distillation from a Fine-Tuned Teacher

If Approach 2 still produces insufficient English quality.

**Step 1: Fine-tune a teacher model.**
- Take a pretrained model with strong English fluency (e.g., Qwen3 0.6B)
- Fine-tune on the domain corpus
- Result: a model that speaks fluent English and knows Wii content, but uses a standard transformer architecture (no PLE)

**Step 2: Generate soft targets.**
- Run the teacher over the training corpus
- For each token position, save the teacher's output logit distribution (top-64 or top-128 logits to save disk)
- Alternatively, run the teacher on-the-fly during training (more GPU time, no extra disk)

**Step 3: Train the PLE student with distillation loss.**
- Loss: `L = alpha * CE(student, ground_truth) + (1 - alpha) * KL(student, teacher_logits)`
- alpha around 0.3-0.5 (tunable)
- Temperature scaling on the KL term (T=2-4 typical)
- The soft targets transfer the teacher's language fluency to the student despite the architectural difference

**What changes vs Approach 2:** addition of a teacher fine-tuning step and a modified loss function. The student training loop now consumes both ground truth tokens and teacher logit distributions.

**What stays the same:** data pipeline, tokenizer, student model architecture, export process.

**Practical cost:**
- Teacher fine-tuning: ~2-4 hours on the 5080
- Soft target generation (if offline): ~2-3 hours for 500M tokens at 0.6B params
- Student training: same as Approach 1/2 plus ~30% overhead for the KL loss computation
- Total: roughly 2x the wall-clock time of Approach 1

## Shared Infrastructure

### Data Pipeline

Common to all approaches:

1. **Collection:** scrape/download domain sources, general English subsets
2. **Cleaning:** deduplication, HTML stripping, encoding normalization, quality filtering
3. **Tokenization:** encode all text with the trained SentencePiece model
4. **Packing:** concatenate tokenized documents with separator tokens, chunk into 512-token sequences
5. **Shuffling:** shuffle at the document level before packing, then shuffle packed sequences

### Export

1. Quantize all weights to INT8 (matching MODEL.md inference spec)
2. Export as raw INT8 tensors with a simple header (no GGUF or other complex format)
3. PLE table layout: `[token_id][layer_0 (384 bytes) | layer_1 | ... | layer_19]` (contiguous per token, one seek per token at inference)
4. Dense core and embedding weights in layer-sequential order for single-pass loading at boot
5. Weight tying: export embedding matrix once, used as both input embedding and lm_head (transposed)

## Evaluation

### Quantitative

- **Perplexity** on held-out domain data (10% random split)
- **Perplexity** on general English held-out set (measures fluency retention)
- **Token accuracy** (top-1 next token prediction accuracy on domain test set)

### Qualitative

- **Fluency:** does the model produce grammatical, coherent English sentences?
- **Domain accuracy:** does it answer Wii trivia questions correctly?
- **Off-topic handling:** does it gracefully degrade on non-Wii questions (vague/repetitive output rather than confident hallucination)?
- **Conversational coherence:** can it sustain multi-turn dialogue within the 512-token context?

### Escalation Criteria

| Symptom | Action |
|---|---|
| Minor grammar/fluency issues | Move to Approach 1.5 (add general English blend) |
| English still weak after blending | Move to Approach 2 (two-phase pretraining) |
| English improved but still noticeably worse than a fine-tuned standard model | Move to Approach 3 (distillation) |
| Both fluency and knowledge are acceptable | Ship it |

### Post-Quantization Validation

After INT8 export, re-evaluate perplexity and qualitative outputs to confirm quantization does not degrade quality beyond acceptable limits. The model is trained in bf16/fp32 but deployed in INT8; some degradation is expected and should be measured.
