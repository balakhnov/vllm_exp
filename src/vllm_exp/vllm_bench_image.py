import time
import torch
import requests
from io import BytesIO
from PIL import Image
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-VL-4B-Instruct"
# MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=4096,
    limit_mm_per_prompt={"image": 1},
    gpu_memory_utilization=0.8,
    enforce_eager=False,
    attention_backend="TRITON_ATTN"
)

tokenizer = llm.get_tokenizer()

# Load image from URL
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=200,
)

inputs = [{
    "prompt_token_ids": prompt_token_ids,
    "multi_modal_data": {"image": image},
}]

# Warmup
for _ in range(10):
    _ = llm.generate(
        inputs,
        sampling_params,
        use_tqdm=False,
    )

torch.cuda.synchronize()

times = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = llm.generate(
        inputs,
        sampling_params,
        use_tqdm=False,
    )

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(f"prompt tokens: {len(prompt_token_ids)}")
print(f"mean: {sum(times)/len(times):.2f} ms")
print(f"min:  {min(times):.2f} ms")
print(f"p50:  {sorted(times)[len(times)//2]:.2f} ms")
print(f"p95:  {sorted(times)[int(len(times)*0.95)]:.2f} ms")