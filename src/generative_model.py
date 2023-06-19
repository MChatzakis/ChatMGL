from trl import PPOTrainer, PPOConfig
from trl.core import LengthSampler
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler

torch.backends.cudnn.deterministic = True
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

from reward_model import *


class GenerativeModel:
    def __init__(self, path, device="cpu"):
        model_name = path
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.tokenizer = GPT2Tokenizer.from_pretrained(path, padding_side="left")
        self.device = device

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(device)

    def generate(
        self,
        question: str,
        max_new_tokens=300,
        do_sample=True,
        num_beams=4,
        length_penalty=1.5,
    ):
        input_ids = self.tokenizer(
            question,
            return_tensors="pt",
            max_length=500,
            truncation=True,
            padding="max_length",
        ).input_ids

        args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            # "top_k": 0,
            # "top_p": 0.9,
            # "temperature": 0.8,
            "length_penalty": length_penalty,
            "num_return_sequences": 1,
        }

        outputs = self.model.generate(input_ids.to(self.device), **args)

        answer = self.tokenizer.decode(
            outputs[0].to(self.device), skip_special_tokens=True
        )
        return answer


def training_RLHF(
    train_dataset,
    model,
    baseline_model,
    reward_model,
    tokenizer,
    dev_dataset=None,
    epochs=1,
    batch_size=4,
    model_save_root="../models/RLHF/",
):
    config = PPOConfig(batch_size=4)

    ppo_trainer = PPOTrainer(
        config, model, baseline_model, tokenizer, dataset=train_dataset
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "batch_size": 1,
    }

    output_min_length = 32
    output_max_length = 400
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
    )

    total_stats = []
    for epoch in range(epochs):
        for batch in tqdm(train_dataloader):
            query_tensors = [
                batch["question_input_ids"][i][0][512:] for i in range(batch_size)
            ]

            #### Get response from model
            response_tensors = []
            for query in query_tensors:
                gen_len = output_length_sampler()
                generation_kwargs["max_new_tokens"] = gen_len
                response = ppo_trainer.generate(query, **generation_kwargs)
                response_tensors.append(response.squeeze()[-gen_len:])
            batch["response"] = [
                tokenizer.decode(r.squeeze()) for r in response_tensors
            ]

            #### Compute reward
            texts = [
                "Human:" + q + "Assistant:" + r
                for q, r in zip(batch["question"], batch["response"])
            ]
            rewards = [reward_model.get_reward(text) for text in texts]

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            total_stats.append(stats)

    model.save_pretrained(model_save_root)
    return total_stats
