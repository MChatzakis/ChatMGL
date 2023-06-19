import random
import numpy as np

from transformers import AdamW, get_constant_schedule_with_warmup
from transformers import get_constant_schedule_with_warmup

from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import numpy as np

from torchmetrics.functional import f1_score

torch.backends.cudnn.deterministic = True

from transformers import PreTrainedModel, PretrainedConfig

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ClassificationRewardModelConfig(PretrainedConfig):
    """
    This is an example config class for a custom HuggingFace model.

    - It currently inherits from the DebertaV2Config class,
    because we are using the OpenAssistant Dberta model as our base model.

    - You are not expected to follow this example, but you can use it as a reference point.
    Inherit from the HuggingFace config class that is most similar to your base model.

    - Or, if you prefer, construct your own config class from scratch if you
    implement your base model from scratch.

    - You should specify the model_type as your model's class name.
    When loading the
    """

    model_type = "ClassificationRewardModel"
    
    # If you have additional parameters to the model class,
    # you can add them inside the config class as well.
    # For example, with "def __init__(self, config, reward_dim=1):",
    # you can specify "reward_dim = 1" here in the config class.
    # Then, you can acess the reward_dim parameter in the model class
    # by calling "self.config.reward_dim".


class ClassificationRewardModel(PreTrainedModel):
    # Set the config class to your custom config class
    config_class = ClassificationRewardModelConfig

    def __init__(self, config, path):
        super().__init__(config)

        # Initialize the base model and its associated tokenizer
        # Absolute path to the pretrained model (GPT2)
        hf_pretrained_model_path = path#"/content/drive/MyDrive/project-m3-chatmgl/models/reward_model/gpt2/lr5e-05-warmup0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_pretrained_model_path
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Set the config
        self.config = config

        # Set the label2id and id2label mappings for the model config
        self.model.config.label2id = {"negative": 0, "positive": 1}
        self.model.config.id2label = {0: "negative", 1: "positive"}

    def forward(self, encoded):
        return self.model(**encoded)

    def predict_class(self, encoded):
        # Check this again, I am not sure.

        # Get the logits
        logits = self.model(**encoded).logits
        class_index = torch.argmax(logits, dim=1)

        # Sanity check
        assert class_index in [0, 1], "Class is not 0 or 1"

        # Return the class label
        return self.model.config.id2label[class_index]

    def get_rewards(self, demonstrations):
        """
        Get the rewards for the demonstrations
        TODO: This is an example function, replace this with your actual implementation!
              Your implementation should handle the input and output format as specified below.

        Args:
            demonstrations: list of dicts in the format of
            {'chosen': str, 'rejected': str}
        Return:
            rewards: list of dicts in the format of
            {'chosen': float, 'rejected': float}
        """
        rewards = []
        for pair in demonstrations:
            # encoded_chosen = self.tokenizer(
            #    pair['chosen'], return_tensors="pt")
            # encoded_reject = self.tokenizer(
            #    pair['rejected'], return_tensors="pt")

            encoded_chosen = self.tokenizer.encode_plus(
                pair["chosen"],
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
            )

            encoded_reject = self.tokenizer.encode_plus(
                pair["rejected"],
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
            )

            scores_chosen_logits = self.forward(encoded_chosen).logits
            scores_reject_logits = self.forward(encoded_reject).logits

            score_chosen = self.get_predicted_class_prob(scores_chosen_logits)
            score_reject = self.get_predicted_class_prob(scores_reject_logits)

            rewards.append({"chosen": score_chosen, "rejected": score_reject})

        return rewards

    def get_predicted_class_prob(self, logits):
        """
        Get the predicted class probability from the logits
        """
        # Get probabilities
        probabilities = F.softmax(logits, dim=1)

        # Get index of the predicted class
        predicted_class_index = torch.argmax(probabilities, dim=1)

        # Get the probability of the predicted class
        return probabilities[0][predicted_class_index].item()


    def get_reward(self, demonstration):
            """
            Returns a number in the range [-0.5,0.5] which corresponds 
            to the shifted probability of being positive.
            
            Args:
                demonstration: str
            Return:
                reward: Torch.Tensor
            """
            rewards = []

            encoded_text = self.tokenizer.encode_plus(
                demonstration,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
            )
            
            with torch.no_grad():
              logits = self.forward(encoded_text).logits

            probs = F.softmax(logits,dim=1)[0][1] - 0.5

            return probs
