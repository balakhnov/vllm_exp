import time
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-VL-4B-Instruct"

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    limit_mm_per_prompt={"image": 0, "video": 0},
    gpu_memory_utilization=0.9,
    enforce_eager=False,
)

tokenizer = llm.get_tokenizer()

messages = [
    {"role": "user", "content": "Describe speculative decoding in one sentence."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1,
)

# Warmup: important for torch.compile / cudagraph capture
for _ in range(10):
    _ = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        sampling_params,
        use_tqdm=False,
    )

torch.cuda.synchronize()

times = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    _ = llm.generate(
        [{"prompt_token_ids": prompt_token_ids}],
        sampling_params,
        use_tqdm=False,
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

print(f"prompt tokens: {len(prompt_token_ids)}")
print(f"mean: {sum(times)/len(times):.2f} ms")
print(f"min:  {min(times):.2f} ms")
print(f"p50:  {sorted(times)[len(times)//2]:.2f} ms")
print(f"p95:  {sorted(times)[int(len(times)*0.95)]:.2f} ms")