conda env remove -n llama.cpp -y
conda env create -n llama.cpp --file environment.yaml
conda run -n llama.cpp pip install torch numpy sentencepiece

cmake .
cmake --build . --config Release

copy bin\Release\embedding.exe embedding.exe
copy bin\Release\main.exe main.exe
copy bin\Release\perplexity.exe perplexity.exe
copy bin\Release\quantize.exe quantize.exe

conda run -n llama.cpp python convert-pth-to-ggml.py ../data/llama/7B/ 1
quantize ../data/llama/7B/ggml-model-f16.bin ../data/llama/7B/ggml-model-q4_0.bin 2

conda run -n llama.cpp python convert-pth-to-ggml.py ../data/llama/13B/ 1
quantize ../data/llama/13B/ggml-model-f16.bin ../data/llama/13B/ggml-model-q4_0.bin 2
