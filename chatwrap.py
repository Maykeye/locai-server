from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from pathlib import Path
from dataclasses import dataclass
import time
import json
import torch

@dataclass
class OngoingGeneration:
    starting_prompt: str
    last_prompt: str
    result: str
    settings: ExLlamaV2Sampler.Settings
    steps: int = 0

    @property 
    def newly_generated(self):
        return self.result[len(self.last_prompt):]

    @property
    def total_generation(self):
        return self.result[len(self.starting_prompt):]


class ChatWrapExl2:
    def __init__(self, p:Path|str) -> None:
        if isinstance(p, str):
            p = Path(p)
        if str(p).startswith("~"):
            p = p.expanduser()
        t0 = time.time()
        assert p.exists(), "Path doest not exist"
        config = ExLlamaV2Config()
        config.model_dir = str(p)
        config.prepare()
        config.max_seq_len = 32768
        self.config = config
        self.model = ExLlamaV2(config)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.cache = ExLlamaV2Cache_Q4(self.model, lazy=True)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)
        self.model.load_autosplit(self.cache)
        print(f"loaded {p.name} in {time.time()-t0:.2f} secs")

    def generate(self, prompt: str, n_tokens=16, return_new_only:bool=False) -> str:
        settings = ExLlamaV2Sampler.Settings()
        text = self.generator.generate_simple(prompt, settings, n_tokens)
        if return_new_only:
            text = text[len(prompt):]
        return text # type: ignore

    def generate_steps(self, prompt, step_size=1):
        settings = ExLlamaV2Sampler.Settings()
        gen = OngoingGeneration(starting_prompt=prompt, last_prompt=prompt, result="", settings=settings)
        while True:
            text = self.generator.generate_simple(prompt, settings, step_size)
            assert isinstance(text, str)
            gen.last_prompt = prompt
            gen.result = text
            gen.steps += step_size
            yield gen
            prompt = gen.result

    def generate_stream(self, initial_prompt: str, step_size: int=16):
        generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        input_ids = self.tokenize(initial_prompt)
        
        settings = ExLlamaV2Sampler.Settings()
        generator.begin_stream(input_ids, settings)
        while True:
           c = generator.stream()
           batch = c[0]
           yield batch

    def tokenize(self, s:str)->torch.Tensor:
        return self.tokenizer.encode(s, return_offsets=False) # type:ignore (we specify return_offsets=False)

    def count_tokens(self, s: str) -> int:
        return self.tokenize(s).shape[-1]

def codegen_prompt(code:str, line_no:int, col_no:int, _cache = {}, tokenizer=None):
    # Note: Not even trying to handle out of bounds
    N=5
    if not _cache:
        _cache['prompt'] = Path('prompt').read_text()
    prompt = _cache['prompt']
    
    lines = code.splitlines()
    cursor_line = lines[line_no]
    lines[line_no] = lines[line_no][:col_no]
    context_lines = lines[max(0, line_no-N):line_no]    
    lines[line_no] = "<|CURSOR|>"+lines[line_no]
    # TODO: clean
    code = "\n".join(lines)
    context = "\n".join(context_lines)
    prompt = prompt.format(CODE=code, CURSORLINE=context)

    del cursor_line
    del tokenizer
    # TODO: insert "safe" tokens (ie everything until space/dot/comma/etc)

    return prompt

def test():
    m = ChatWrapExl2("~/models/bartowski_zephyr-7b-dpo-full-exl2_4_25")
    t = (codegen_prompt("imp", 0, 3))
    print(t)
    print(t, end="|>")
    t = m.generate(t, return_new_only=True)
    print(t)

def test1():
    m = ChatWrapExl2("~/models/bartowski_zephyr-7b-dpo-full-exl2_4_25") # This works the best from OSS models(MIT/Apache2) what I've tried 
    t = codegen_prompt("""def calculate_sum(lst):
    rv = 0.0
    for i
""".strip(), 2, 8)
    print(t)
    print(t, end="|>")
    t = m.generate(t, return_new_only=True)
    print(t)

if __name__ == "__main__":
    test()
