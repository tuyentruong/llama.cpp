#include <iostream>
#include "common.h"

int generate_embedding(std::string prompt, gpt_params *params, llama_context * ctx) {
    int n_past = 0;
    // Add a space in front of the first character to match OG llama tokenizer behavior
    prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, prompt, true);

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params->verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i : embd_inp) {
            fprintf(stderr, "%6d -> '%s'\n", i, llama_token_to_str(ctx, i));
        }
        fprintf(stderr, "\n");
    }

    if (params->embedding){
        if (!embd_inp.empty()) {
            if (llama_eval(ctx, embd_inp.data(), (int) embd_inp.size(), n_past, params->n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        const int n_embd = llama_n_embd(ctx);
        const auto embeddings = llama_get_embeddings(ctx);

        for (int i = 0; i < n_embd; i++) {
            printf("%f ", embeddings[i]);
        }
        printf("\n");
    }

    return 0;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (!gpt_params_parse(argc, argv, params)) {
        return 1;
    }

    params.embedding = true;

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_context * ctx;

    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.logits_all = params.perplexity;
        lparams.use_mlock  = params.use_mlock;
        lparams.embedding  = params.embedding;

        ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == nullptr) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }
    int exit_code = 0;
    if (params.stream) {
        fprintf(stderr, "generating embeddings from stream\n");
        std::string prompt;
        while (std::getline(std::cin, prompt)) {
            if (prompt.empty()) {
                fprintf(stderr, "received empty prompt: quitting\n");
                fflush(stderr);
                break;
            }
            generate_embedding(prompt, &params, ctx);
        }
    } else {
        // generate a single sample
        exit_code = generate_embedding(params.prompt, &params, ctx);
    }

    llama_print_timings(ctx);
    llama_free(ctx);

    return exit_code;
}

