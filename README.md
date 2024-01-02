# State of What Art? A Call for Multi-Prompt LLM Evaluation

This repository contains the data for the paper: http://arxiv.org/abs/2401.00595

Authors: Moran Mizrahi, Guy Kaplan, Dan Malkin, Rotem Dror, Dafna Shahaf, Gabriel Stanovsky.
The Hebrew University of Jerusalem, Israel. 

## Instruction template paraphrases

For access to the paraphrase datasets used in our project, please explore the `automatic paraphrases` folder. This folder contains all automatically generated paraphrases for the LMentry and BBH benchmarks. Within this category, you will find a CSV file for each task we evaluated in these benchmarks. These files provide the paraphrases along with additional information such as the method used for generating the paraphrase, a manual assessment of its correctness, and the length of the paraphrase measured in tokens.

## Model Accuracies

The `model accuracies` folder provides detailed information on the computed accuracies of various models for each paraphrase. For each task, you will find a dedicated CSV file. In this CSV file, each row corresponds to a task paraphrase, identified by its index. The columns represent different models. For each model and paraphrase pair, the file details the model's accuracy based on 100 samples.

## Aggregated Results

Within the `aggregated results (jsons)` folder, you will find the results for each task formatted as JSON files. Each entry in these files is structured as follows:

```json
"4": {
  "template": "Is {category} the category for all {words}? Respond with either \"yes\" or \"no\".",
  "method": "rephrase",
  "correct": true,
  "accs": {
    "t0_3b": 0.58,
    "t0pp": 0.96,
    ...
  },
  "ranks": {
    "t0_3b": 6,
    "t0pp": 1,
    ...
  }
}
```
In this structure, each key represents the index of a paraphrase. The **template** key provides the instruction template, **method** indicates the technique used for generating the paraphrase, and **correct** is a boolean value reflecting the manual assessment of the paraphrase's correctness. The **accs** key contains a dictionary of model accuracies for the respective paraphrase, and **ranks** presents the rankings of these models based on their accuracies for that specific paraphrase. 

## Raw Model Responses

Finally, for comprehensive access to all model responses, a zipped folder is available for download. The size of the zipped file is approximately 190 MB. You can download it using the following link: [Download Raw Model Responses](https://www.dropbox.com/scl/fo/y9dd8zbteyf0xrjxdtm3e/h?rlkey=okp52gleuibw72fhe62egr6lp&dl=0). Please note that the unzipped data expands to around 4GB. 

Enjoy! :-)
