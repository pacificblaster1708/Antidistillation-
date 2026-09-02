"""tests/make_fixtures.py -- build tiny local fixtures: two *different* byte-level BPE tokenizers, tiny
Qwen2/Llama models, and a repo-format traces dataset. No network needed."""
import json, os, random, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIXTURES  # noqa: E402
import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                          LlamaConfig, LlamaForCausalLM)
import datasets as hfds

ROOT = FIXTURES
random.seed(0)

WORDS = ("the a of to and in is are we have step so let compute answer result sum "
         "product number value equals therefore final boxed problem solution first "
         "second third multiply divide add subtract total cost price apples oranges "
         "John Mary bought sold each day week hour minute dollars cents").split()

def corpus(n, seed):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        k = r.randint(8, 40)
        s = " ".join(r.choice(WORDS) for _ in range(k))
        s += f" {r.randint(1,999)} * {r.randint(1,99)} = {r.randint(1,9999)}.\n"
        if r.random() < 0.4:
            s += "\\boxed{" + str(r.randint(0, 500)) + "}\n"
        out.append(s)
    return out

def build_bpe(vocab_size, seed, path, chat_template, extra_specials=()):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    specials = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"] + list(extra_specials)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False, min_frequency=1)
    tok.train_from_iterator(corpus(4000, seed), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        bos_token=None,
        unk_token=None,
    )
    fast.chat_template = chat_template
    os.makedirs(path, exist_ok=True)
    fast.save_pretrained(path)
    return fast

QWEN_TMPL = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
LLAMA_TMPL = (
    "{% for m in messages %}<|start|>{{ m['role'] }}\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|start|>assistant\n{% endif %}"
)

def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(ROOT, exist_ok=True)

    # --- tokenizers: qwen-like (teacher + shared student) and llama-like -----
    tk_q = build_bpe(1000, seed=1, path=f"{ROOT}/tok_qwen", chat_template=QWEN_TMPL)
    tk_l = build_bpe(900, seed=2, path=f"{ROOT}/tok_llama", chat_template=LLAMA_TMPL,
                     extra_specials=["<|start|>", "<|end|>"])
    print("qwen tok size", len(tk_q), "llama tok size", len(tk_l))
    probe = "the answer is 42 apples and the total cost"
    same = tk_q(probe)["input_ids"] == tk_l(probe)["input_ids"]
    assert not same, "fixture tokenizers must disagree -- the cross-vocab tests depend on it"
    print("tokenizers genuinely disagree:", not same)

    # --- models -------------------------------------------------------------
    def qwen(vocab, path):
        cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=2048, tie_word_embeddings=False)
        m = Qwen2ForCausalLM(cfg)
        m.save_pretrained(path); return m

    def llama(vocab, path):
        cfg = LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=2048, tie_word_embeddings=False)
        m = LlamaForCausalLM(cfg)
        m.save_pretrained(path); return m

    qwen(1200, f"{ROOT}/teacher_qwen")        # "7B" stand-in, padded vocab 1200
    qwen(1100, f"{ROOT}/student_qwen")        # "3B" shared-vocab student, vocab 1100
    llama(900, f"{ROOT}/student_llama")       # "3B" cross-tokenizer student
    # Larger student with enough capacity to actually fit the teacher --
    # used by test_learning.py to prove the objective converges.
    cfg_big = Qwen2Config(vocab_size=1100, hidden_size=256, intermediate_size=512,
                          num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
                          max_position_embeddings=2048, tie_word_embeddings=False)
    Qwen2ForCausalLM(cfg_big).save_pretrained(f"{ROOT}/student_qwen_big")

    # --- repo-format traces --------------------------------------------------
    r = random.Random(7)
    rows = []
    for i in range(24):
        problem = f"John has {r.randint(2,50)} apples and buys {r.randint(2,20)} more. How many apples?"
        answer = r.randint(2, 70)
        body = (" ".join(r.choice(WORDS) for _ in range(r.randint(20, 60)))
                + f"\nStep 1: add the numbers.\nStep 2: total is {answer}.\n"
                + "\\boxed{" + str(answer) + "}")
        trace = f"<｜User｜>{problem}<｜Assistant｜>{body}<｜end▁of▁sentence｜>"
        rows.append({"problem": problem, "trace": trace, "solution": str(answer)})
    ds = hfds.Dataset.from_list(rows)
    ds.save_to_disk(f"{ROOT}/traces")
    ds.select(range(6)).save_to_disk(f"{ROOT}/holdout")

    with open(f"{ROOT}/pairs.jsonl", "w") as f:
        for row in rows[:8]:
            f.write(json.dumps({"prompt": row["problem"],
                                "completion": row["trace"].split("<｜Assistant｜>")[1]
                                .replace("<｜end▁of▁sentence｜>", "")}) + "\n")
    print("fixtures written to", ROOT)

if __name__ == "__main__":
    main()
