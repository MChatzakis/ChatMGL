import json
import json
from statistics import mean
from evaluate import load
import nltk


def get_supervised_data(filename):
    # open the file
    with open(filename, "r") as f:
        data = json.load(f)

    # take the correct interaction and split out the human and assistant part
    pairs = []  # list of interaction pairs [(human,assistant),....]
    for entry in data:
        question = entry["question"]
        answer   = entry["answer"]
        if "choices" in entry:
          if entry['choices'] is not None:
            choices = ' '.join(entry["choices"])
            question += choices
        if isinstance(answer,list):
          answer = ' '.join(answer)
        pairs.append((question, answer))

    return pairs


def read_json(filename):
    with open(filename, "r") as data_file:
        return json.load(data_file)


def get_bert_scores(predictions, labels, model_type="distilbert-base-uncased"):
    """Returns BERTScore {f1, precision, recall} for the given predictions and labels.

    Args:
        predictions (_type_): _description_
        labels (_type_): _description_
        model_type (str, optional): _description_. Defaults to "distilbert-base-uncased".

    Returns:
        _type_: _description_
    """
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=predictions, references=labels, model_type=model_type
    )
    # print(results)
    return mean(results["f1"]), mean(results["recall"]), mean(results["precision"])


def get_rouge_scores(predictions, labels):
    """Returns ROUGE {rouge1, rouge2, rougeL, rougeLsum} for the given predictions and labels.

    Args:
        predictions (_type_): _description_
        labels (_type_): _description_

    Returns:
        _type_: _description_
    """
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=labels)
    # print(results)
    return results["rouge1"], results["rouge2"], results["rougeL"], results["rougeLsum"]


def get_bleu_scores(predictions, labels):
    bleu = []
    bleu1 = []
    bleu4 = []
    brevity = []

    for pred, ref in zip(predictions, labels):
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        # Create a list of reference sentences
        ref_sentences = [ref_tokens]

        # Create a list of predicted sentences
        pred_sentences = pred_tokens

        # Calculate BLEU score with 4-gram maximum
        bleu_score = nltk.translate.bleu_score.sentence_bleu(
            ref_sentences, pred_sentences
        )
        bleu1_score = nltk.translate.bleu_score.sentence_bleu(
            ref_sentences, pred_sentences, weights=(1, 0, 0, 0)
        )
        bleu4_score = nltk.translate.bleu_score.sentence_bleu(
            ref_sentences, pred_sentences, weights=(0, 0, 0, 1)
        )
        brevity_penalty = nltk.translate.bleu_score.brevity_penalty(
            len(ref_tokens), len(pred_tokens)
        )

        bleu.append(bleu_score)
        bleu1.append(bleu1_score)
        bleu4.append(bleu4_score)
        brevity.append(brevity_penalty)

    return mean(bleu), mean(bleu1), mean(bleu4), mean(brevity)

