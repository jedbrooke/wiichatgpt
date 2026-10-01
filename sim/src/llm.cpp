// Llama 2 inference engine for WiiChatGPT.
// Loads INT8 quantized model, uses wii_matvec_q8 for fused dequant+compute.
// Small vectors (embeddings, norm weights) dequantized inline via scratch.
//
// Ported from karpathy/llama2.c run.c with INT8 weight loading
// and wii_math.h abstraction layer.

#include "llm.h"
#include "wii_math.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>

// ----------------------------------------------------------------------------
// Dequantization (for small vectors only: embeddings, norm weights)

static void dequantize(const int8_t* q, float scale, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (float)q[i] * scale;
    }
}

// Dequantize a slice of a quantized tensor into the scratch buffer.
// Only for small vectors (dim-sized), not for weight matrices.
static float* dequant_to_scratch(LlmModel* model, const QuantizedTensor* qt, int offset, int count) {
    dequantize(qt->data + offset, qt->scale, model->state.scratch, count);
    return model->state.scratch;
}

// ----------------------------------------------------------------------------
// Model loading

static QuantizedTensor read_qtensor(uint8_t** cursor, int size) {
    QuantizedTensor qt;
    qt.size = size;
    qt.scale = *(float*)(*cursor);
    *cursor += sizeof(float);
    qt.data = (const int8_t*)(*cursor);
    *cursor += size;
    return qt;
}

int llm_load_model(LlmModel* model, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "llm_load_model: cannot open %s\n", path);
        return -1;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    model->raw_data_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read entire file into memory
    model->raw_data = (uint8_t*)malloc(model->raw_data_size);
    if (!model->raw_data) {
        fprintf(stderr, "llm_load_model: malloc failed (%zu bytes)\n", model->raw_data_size);
        fclose(f);
        return -1;
    }
    size_t read_bytes = fread(model->raw_data, 1, model->raw_data_size, f);
    fclose(f);
    if (read_bytes != model->raw_data_size) {
        fprintf(stderr, "llm_load_model: short read (%zu / %zu)\n", read_bytes, model->raw_data_size);
        free(model->raw_data);
        return -1;
    }

    // Parse config header
    uint8_t* cursor = model->raw_data;
    memcpy(&model->config, cursor, sizeof(LlmConfig));
    cursor += sizeof(LlmConfig);

    LlmConfig* p = &model->config;
    model->weights.shared_weights = p->vocab_size > 0 ? 1 : 0;
    p->vocab_size = abs(p->vocab_size);

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int n_layers = p->n_layers;
    int n_heads = p->n_heads;
    int n_kv_heads = p->n_kv_heads;
    int vocab_size = p->vocab_size;
    int head_size = dim / n_heads;
    int kv_dim = (dim * n_kv_heads) / n_heads;

    fprintf(stderr, "Model: dim=%d hidden_dim=%d n_layers=%d n_heads=%d n_kv_heads=%d vocab=%d seq_len=%d\n",
            dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, p->seq_len);

    // Parse quantized tensors
    LlmWeights* w = &model->weights;
    w->token_embedding_table = read_qtensor(&cursor, vocab_size * dim);
    w->rms_att_weight        = read_qtensor(&cursor, n_layers * dim);
    w->wq                    = read_qtensor(&cursor, n_layers * dim * (n_heads * head_size));
    w->wk                    = read_qtensor(&cursor, n_layers * dim * kv_dim);
    w->wv                    = read_qtensor(&cursor, n_layers * dim * kv_dim);
    w->wo                    = read_qtensor(&cursor, n_layers * (n_heads * head_size) * dim);
    w->rms_ffn_weight        = read_qtensor(&cursor, n_layers * dim);
    w->w1                    = read_qtensor(&cursor, n_layers * dim * hidden_dim);
    w->w2                    = read_qtensor(&cursor, n_layers * hidden_dim * dim);
    w->w3                    = read_qtensor(&cursor, n_layers * dim * hidden_dim);
    w->rms_final_weight      = read_qtensor(&cursor, dim);

    if (w->shared_weights) {
        w->wcls = w->token_embedding_table;
    } else {
        w->wcls = read_qtensor(&cursor, vocab_size * dim);
    }

    // Allocate run state
    LlmRunState* s = &model->state;
    s->x = (float*)calloc(dim, sizeof(float));
    s->xb = (float*)calloc(dim, sizeof(float));
    s->xb2 = (float*)calloc(dim, sizeof(float));
    s->hb = (float*)calloc(hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(hidden_dim, sizeof(float));
    s->q = (float*)calloc(dim, sizeof(float));
    s->k = (float*)calloc(kv_dim, sizeof(float));
    s->v = (float*)calloc(kv_dim, sizeof(float));
    s->att = (float*)calloc(n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(vocab_size, sizeof(float));
    s->key_cache = (float*)calloc(n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(n_layers * p->seq_len * kv_dim, sizeof(float));

    // Small scratch buffer for dequanting vectors (norm weights, embeddings)
    int scratch_size = dim > hidden_dim ? dim : hidden_dim;
    s->scratch = (float*)calloc(scratch_size, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q ||
        !s->k || !s->v || !s->att || !s->logits ||
        !s->key_cache || !s->value_cache || !s->scratch) {
        fprintf(stderr, "llm_load_model: allocation failed\n");
        llm_free_model(model);
        return -1;
    }

    size_t consumed = (size_t)(cursor - model->raw_data);
    fprintf(stderr, "Loaded %zu bytes of %zu (%.1f MB model)\n",
            consumed, model->raw_data_size, (double)consumed / (1024.0 * 1024.0));

    return 0;
}

void llm_free_model(LlmModel* model) {
    LlmRunState* s = &model->state;
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
    free(s->scratch);
    memset(s, 0, sizeof(LlmRunState));

    free(model->raw_data);
    model->raw_data = nullptr;
}

// ----------------------------------------------------------------------------
// Forward pass

float* llm_forward(LlmModel* model, int token, int pos) {
    LlmConfig* p = &model->config;
    LlmWeights* w = &model->weights;
    LlmRunState* s = &model->state;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;

    float* x = s->x;

    // Token embedding lookup: dequantize the row for this token
    dequantize(w->token_embedding_table.data + token * dim,
               w->token_embedding_table.scale, x, dim);

    // Forward all layers
    for (int l = 0; l < p->n_layers; l++) {

        // Attention RMSNorm
        {
            float* rms_w = dequant_to_scratch(model, &w->rms_att_weight, l * dim, dim);
            wii_rmsnorm(x, rms_w, s->xb, dim);
        }

        // KV cache pointers for this layer and position
        int loff = l * p->seq_len * kv_dim;
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* val_cache_row = s->value_cache + loff + pos * kv_dim;

        // QKV matmuls (fused dequant via wii_matvec_q8)
        wii_matvec_q8((const signed char*)(w->wq.data + l * dim * dim),
                      w->wq.scale, s->xb, s->q, dim, dim);
        wii_matvec_q8((const signed char*)(w->wk.data + l * dim * kv_dim),
                      w->wk.scale, s->xb, key_cache_row, kv_dim, dim);
        wii_matvec_q8((const signed char*)(w->wv.data + l * dim * kv_dim),
                      w->wv.scale, s->xb, val_cache_row, kv_dim, dim);

        // RoPE: rotate q and k
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : key_cache_row;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // Multi-head attention
        for (int h = 0; h < p->n_heads; h++) {
            float* q_head = s->q + h * head_size;
            float* att_head = s->att + h * p->seq_len;

            // Compute attention scores: q dot k for all positions up to pos
            for (int t = 0; t <= pos; t++) {
                float* k_t = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_t[i];
                }
                att_head[t] = score / sqrtf((float)head_size);
            }

            // Softmax over attention scores
            wii_softmax(att_head, att_head, pos + 1);

            // Weighted sum of values
            float* xb_head = s->xb + h * head_size;
            memset(xb_head, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v_t = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att_head[t];
                for (int i = 0; i < head_size; i++) {
                    xb_head[i] += a * v_t[i];
                }
            }
        }

        // Output projection (fused dequant)
        wii_matvec_q8((const signed char*)(w->wo.data + l * dim * dim),
                      w->wo.scale, s->xb, s->xb2, dim, dim);

        // Residual connection
        wii_add(x, s->xb2, x, dim);

        // FFN RMSNorm
        {
            float* rms_w = dequant_to_scratch(model, &w->rms_ffn_weight, l * dim, dim);
            wii_rmsnorm(x, rms_w, s->xb, dim);
        }

        // FFN: SwiGLU (fused dequant)
        // hb = silu(w1 @ xb) * (w3 @ xb)
        wii_matvec_q8((const signed char*)(w->w1.data + l * dim * hidden_dim),
                      w->w1.scale, s->xb, s->hb, hidden_dim, dim);
        wii_matvec_q8((const signed char*)(w->w3.data + l * dim * hidden_dim),
                      w->w3.scale, s->xb, s->hb2, hidden_dim, dim);

        // SiLU on hb, then elementwise multiply with hb2
        wii_silu(s->hb, s->hb, hidden_dim);
        wii_mul(s->hb, s->hb2, s->hb, hidden_dim);

        // Down projection (fused dequant): xb = w2 @ hb
        wii_matvec_q8((const signed char*)(w->w2.data + l * hidden_dim * dim),
                      w->w2.scale, s->hb, s->xb, dim, hidden_dim);

        // Residual connection
        wii_add(x, s->xb, x, dim);
    }

    // Final RMSNorm
    {
        float* rms_w = dequant_to_scratch(model, &w->rms_final_weight, 0, dim);
        wii_rmsnorm(x, rms_w, x, dim);
    }

    // Classifier: logits = wcls @ x (fused dequant)
    wii_matvec_q8((const signed char*)w->wcls.data,
                  w->wcls.scale, x, s->logits, p->vocab_size, dim);

    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer (BPE, ported from llama2.c)

static int compare_tokens(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int llm_build_tokenizer(LlmTokenizer* t, const char* path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = nullptr;

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "llm_build_tokenizer: cannot open %s\n", path);
        return -1;
    }

    if (fread(&t->max_token_length, sizeof(int), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, f) != 1) { fclose(f); return -1; }
        if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return -1; }
        t->vocab[i] = (char*)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, f) != 1) { fclose(f); return -1; }
        t->vocab[i][len] = '\0';
    }

    fclose(f);
    return 0;
}

void llm_free_tokenizer(LlmTokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

const char* llm_decode(LlmTokenizer* t, int prev_token, int token) {
    char* piece = t->vocab[token];
    // After BOS token (1), strip leading whitespace
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // Handle raw byte tokens like <0x01>
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

static int str_lookup(const char* str, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex tok;
    tok.str = (char*)str;
    tok.id = 0;
    TokenIndex* res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size,
                                            sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

void llm_encode(LlmTokenizer* t, const char* text, int bos, int eos,
                int* tokens, int* n_tokens) {
    if (!text) {
        fprintf(stderr, "llm_encode: null text\n");
        return;
    }

    // Lazily sort vocabulary
    if (!t->sorted_vocab) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = (char*)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    // Add dummy prefix if text is non-empty
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Encode UTF-8 byte sequence
    for (const char* c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    // BPE merge loop
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            snprintf(str_buffer, t->max_token_length * 2 + 1 + 2, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler

static unsigned int random_u32(unsigned long long* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * 0x2545F4914F6CDD1Dull) >> 32);
}

static float random_f32(unsigned long long* state) {
    return (float)(random_u32(state) >> 8) / 16777216.0f;
}

static int sample_argmax(float* probs, int n) {
    int max_i = 0;
    float max_p = probs[0];
    for (int i = 1; i < n; i++) {
        if (probs[i] > max_p) {
            max_i = i;
            max_p = probs[i];
        }
    }
    return max_i;
}

static int sample_mult(float* probs, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int compare_probindex(const void* a, const void* b) {
    const ProbIndex* a_ = (const ProbIndex*)a;
    const ProbIndex* b_ = (const ProbIndex*)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

static int sample_topp(float* probs, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (float)(n - 1);
    for (int i = 0; i < n; i++) {
        if (probs[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probs[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare_probindex);

    float cumulative = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative += probindex[i].prob;
        if (cumulative > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) return probindex[i].index;
    }
    return probindex[last_idx].index;
}

void llm_build_sampler(LlmSampler* s, int vocab_size, float temperature,
                       float topp, unsigned long long rng_seed) {
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng_state = rng_seed;
    s->probindex = (ProbIndex*)malloc(vocab_size * sizeof(ProbIndex));
}

void llm_free_sampler(LlmSampler* s) {
    free(s->probindex);
}

// In-place softmax for sampling (modifies logits)
static void softmax_inplace(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

int llm_sample(LlmSampler* s, float* logits) {
    int next;
    if (s->temperature == 0.0f) {
        next = sample_argmax(logits, s->vocab_size);
    } else {
        for (int q = 0; q < s->vocab_size; q++) {
            logits[q] /= s->temperature;
        }
        softmax_inplace(logits, s->vocab_size);
        float coin = random_f32(&s->rng_state);
        if (s->topp <= 0.0f || s->topp >= 1.0f) {
            next = sample_mult(logits, s->vocab_size, coin);
        } else {
            next = sample_topp(logits, s->vocab_size, s->topp, s->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Timing

static long time_in_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// Safe print (skip non-printable control bytes)

static void safe_printf(const char* piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char b = (unsigned char)piece[0];
        if (!(isprint(b) || isspace(b))) return;
    }
    printf("%s", piece);
}

// ----------------------------------------------------------------------------
// Generation loop

void llm_generate(LlmModel* model, LlmTokenizer* tokenizer,
                  LlmSampler* sampler, const char* prompt, int steps) {
    const char* empty = "";
    if (!prompt) prompt = empty;

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    llm_encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Expected at least 1 prompt token\n");
        free(prompt_tokens);
        return;
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < steps) {
        float* logits = llm_forward(model, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = llm_sample(sampler, logits);
        }
        pos++;

        if (next == 1) break; // BOS = end of sequence

        const char* piece = llm_decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) start = time_in_ms();
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (double)(pos - 1) / (double)(end - start) * 1000.0);
    }

    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// CLI main

#ifndef LLM_NO_MAIN

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s <checkpoint> [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature (default 1.0)\n");
    fprintf(stderr, "  -p <float>  top-p sampling (default 0.9)\n");
    fprintf(stderr, "  -s <int>    random seed (default: time)\n");
    fprintf(stderr, "  -n <int>    max tokens (default 256, 0 = seq_len)\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> tokenizer path (default: tokenizer.bin)\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    const char* prompt = nullptr;
    unsigned long long rng_seed = 0;

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) {
            usage(argv[0]);
            return 1;
        }
        switch (argv[i][1]) {
            case 't': temperature = (float)atof(argv[i + 1]); break;
            case 'p': topp = (float)atof(argv[i + 1]); break;
            case 's': rng_seed = (unsigned long long)atoll(argv[i + 1]); break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'i': prompt = argv[i + 1]; break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            default: usage(argv[0]); return 1;
        }
    }

    if (rng_seed == 0) rng_seed = (unsigned long long)time(nullptr);
    if (temperature < 0.0f) temperature = 0.0f;
    if (topp < 0.0f || topp > 1.0f) topp = 0.9f;
    if (steps < 0) steps = 0;

    wii_math_init();

    LlmModel model;
    memset(&model, 0, sizeof(model));
    if (llm_load_model(&model, checkpoint_path) != 0) {
        return 1;
    }

    if (steps == 0 || steps > model.config.seq_len) {
        steps = model.config.seq_len;
    }

    LlmTokenizer tokenizer;
    memset(&tokenizer, 0, sizeof(tokenizer));
    if (llm_build_tokenizer(&tokenizer, tokenizer_path, model.config.vocab_size) != 0) {
        llm_free_model(&model);
        return 1;
    }

    LlmSampler sampler;
    llm_build_sampler(&sampler, model.config.vocab_size, temperature, topp, rng_seed);

    llm_generate(&model, &tokenizer, &sampler, prompt, steps);

    llm_free_sampler(&sampler);
    llm_free_tokenizer(&tokenizer);
    llm_free_model(&model);
    wii_math_shutdown();

    return 0;
}

#endif // LLM_NO_MAIN