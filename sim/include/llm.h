#pragma once

// Llama 2 inference engine for WiiChatGPT.
// Loads INT8-quantized weights, uses wii_matvec_q8 for fused dequant+compute.
// Small vectors (embeddings, norm weights) are dequantized inline.

#include <cstdint>
#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

// Model config (matches llama2.c binary header: 7 x int32)
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} LlmConfig;

// INT8 quantized weight: per-tensor scale + int8 data
typedef struct {
    float scale;
    const int8_t* data;
    int size;
} QuantizedTensor;

// All model weights (INT8)
typedef struct {
    QuantizedTensor token_embedding_table; // (vocab_size * dim)
    QuantizedTensor rms_att_weight;        // (n_layers * dim)
    QuantizedTensor wq;                    // (n_layers * dim * dim)
    QuantizedTensor wk;                    // (n_layers * dim * kv_dim)
    QuantizedTensor wv;                    // (n_layers * dim * kv_dim)
    QuantizedTensor wo;                    // (n_layers * dim * dim)
    QuantizedTensor rms_ffn_weight;        // (n_layers * dim)
    QuantizedTensor w1;                    // (n_layers * hidden_dim * dim)
    QuantizedTensor w2;                    // (n_layers * dim * hidden_dim)
    QuantizedTensor w3;                    // (n_layers * hidden_dim * dim)
    QuantizedTensor rms_final_weight;      // (dim)
    QuantizedTensor wcls;                  // (vocab_size * dim) or alias to embedding
    int shared_weights;
} LlmWeights;

// Activation buffers and KV cache (all fp32)
typedef struct {
    float* x;           // current activation (dim)
    float* xb;          // activation inside residual branch (dim)
    float* xb2;         // extra buffer (dim)
    float* hb;          // FFN hidden buffer (hidden_dim)
    float* hb2;         // FFN hidden buffer 2 (hidden_dim)
    float* q;           // query (dim)
    float* k;           // key (kv_dim)
    float* v;           // value (kv_dim)
    float* att;         // attention scores (n_heads * seq_len)
    float* logits;      // output logits (vocab_size)
    float* key_cache;   // KV cache keys (n_layers * seq_len * kv_dim)
    float* value_cache; // KV cache values (n_layers * seq_len * kv_dim)
    // Small scratch buffer for dequantizing vectors (dim-sized, e.g. norm weights)
    float* scratch;     // max(dim, hidden_dim)
} LlmRunState;

// Complete model
typedef struct {
    LlmConfig config;
    LlmWeights weights;
    LlmRunState state;
    uint8_t* raw_data;    // raw file data (owns memory)
    size_t raw_data_size;
} LlmModel;

// BPE Tokenizer
typedef struct {
    char* str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} LlmTokenizer;

// Sampler
typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} LlmSampler;

// Model loading
int llm_load_model(LlmModel* model, const char* path);
void llm_free_model(LlmModel* model);

// Forward pass: returns pointer to logits (model->state.logits)
float* llm_forward(LlmModel* model, int token, int pos);

// Tokenizer
int llm_build_tokenizer(LlmTokenizer* t, const char* path, int vocab_size);
void llm_free_tokenizer(LlmTokenizer* t);
void llm_encode(LlmTokenizer* t, const char* text, int bos, int eos,
                int* tokens, int* n_tokens);
const char* llm_decode(LlmTokenizer* t, int prev_token, int token);

// Sampler
void llm_build_sampler(LlmSampler* s, int vocab_size, float temperature,
                       float topp, unsigned long long rng_seed);
void llm_free_sampler(LlmSampler* s);
int llm_sample(LlmSampler* s, float* logits);

// Generation loop
void llm_generate(LlmModel* model, LlmTokenizer* tokenizer,
                  LlmSampler* sampler, const char* prompt, int steps);

#ifdef __cplusplus
}
#endif