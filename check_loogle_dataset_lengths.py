from types import SimpleNamespace

from data_processing import get_dataset
from sglang.benchmark.utils import get_tokenizer


def main():
    model = "/mnt/data/models/Llama-3.1-8B-Instruct"
    tokenizer = get_tokenizer(model)
    for num_prompts in [1, 2, 4, 8]:
        args = SimpleNamespace(
            dataset_name="loogle",
            dataset_path="/mnt/data/dpser/datasets/loogle/longdep_qa_sglang_qa_pairs.jsonl",
            num_prompts=num_prompts,
            disable_shuffle=True,
            enable_multiturn=False,
            enable_shared_prefix=True,
            fixed_output_len=16,
            max_frames=999999,
            model=model,
        )
        data = get_dataset(args, tokenizer)
        prompt_lens = [item[1] for group in data for item in group]
        print(
            "num_prompts",
            num_prompts,
            "groups",
            len(data),
            "requests",
            len(prompt_lens),
            "min",
            min(prompt_lens),
            "max",
            max(prompt_lens),
            "sum",
            sum(prompt_lens),
            "first_group_requests",
            len(data[0]),
        )


if __name__ == "__main__":
    main()
