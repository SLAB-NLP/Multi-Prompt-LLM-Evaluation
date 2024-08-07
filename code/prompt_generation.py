import os
import pandas as pd
import time
import re
import json
import tiktoken
import ast

from call_model import call_model


CGPT_ENCODING_NAME = "cl100k_base"
MODEL_NAME = "cgpt"
DERIVE_PROMPT_MODEL_NAME = "gpt-3.5-turbo-instruct"  # Note: for the paper we used "text-davinci-003", but it is not available anymore
MODEL_TEMPERATURE = 0.3
META_COT_MAX_TOKENS = 500
DEFAULT_MAX_TOKENS = 180

def insert_prompt_row(df, prompt_idx, default_prompt, method, task_description, task_description_source,
                      task_description_length, language_style, meta_prompt, model_name, model_temperature,
                      max_tokens, model_response, prompt_template, prompt_format, prompt_length):

    new_row = {"prompt idx": prompt_idx,
               "default prompt": default_prompt,
               "method": method,
               "task description": task_description,
               "task description source": task_description_source,
               "task description length": task_description_length,
               "language style": language_style,
               "meta prompt": meta_prompt,
               "model name": model_name,
               "model temperature": model_temperature,
               "max tokens": max_tokens,
               "model response": model_response,
               "prompt template": prompt_template,
               "prompt format": prompt_format,
               "prompt length": prompt_length}

    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def get_default_prompts_and_hints(default_prompt_json_path, task_name):

    with open(default_prompt_json_path, 'r', encoding='utf8') as f:
        default_prompts_and_hints = json.load(f)

    return default_prompts_and_hints[task_name]['input_templates'], default_prompts_and_hints[task_name]['hints']


def get_prompt_format(prompt_template):
    if "Q:" in prompt_template and "A:" in prompt_template:
        return "Q&A"
    elif ": {" in prompt_template:
        return "SEP_VARS"
    else:
        return "other"


def get_text_num_tokens(text, encoding_name=CGPT_ENCODING_NAME):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def add_default_prompts(df, default_prompts):
    print("Adding default prompts...")
    for prompt_idx, default_prompt in enumerate(default_prompts):
        df = insert_prompt_row(df=df,
                               prompt_idx=prompt_idx + 1,
                               default_prompt=True,
                               method="default",
                               task_description=None,
                               task_description_length=None,
                               task_description_source=None,
                               language_style=None,
                               meta_prompt=None,
                               model_name=None,
                               model_temperature=None,
                               max_tokens=None,
                               model_response=None,
                               prompt_template=default_prompt,
                               prompt_format=get_prompt_format(default_prompt),
                               prompt_length=get_text_num_tokens(default_prompt))

    return df


# check whether the prompt is valid (contains all the keywords)
def is_prompt_valid(prompt, keywords):
    prompt_variables = set(re.findall('{([^}]+)', prompt))
    if prompt_variables == set(keywords):
        return True
    return False

def correct_prompt(prompt, keywords):
    prompt_variables = set(re.findall('{([^}]+)', prompt))
    # handle the case where the prompt contains unknown variable:
    diff = prompt_variables - set(keywords)
    if diff:
        for prompt_variable in diff:  # remove brackets
            prompt = prompt.replace("{" + prompt_variable + "}", prompt_variable)

    return prompt


def add_rephrase_prompts_lmentry(df, default_prompt_templates, meta_prompt_template, task_variables, num_of_templates_to_generate=1):
    print("Adding rephrase prompts...")
    cur_index = df["prompt idx"].max() + 1
    prompt_templates_all = []

    for default_prompt_template in default_prompt_templates:
        cur_prompt_templates = []
        request = meta_prompt_template.substitute(prompt_template=default_prompt_template)
        cur_temperature = MODEL_TEMPERATURE
        while len(cur_prompt_templates) < (num_of_templates_to_generate / len(default_prompt_templates)):

            response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                                  max_tokens=DEFAULT_MAX_TOKENS, for_creating_prompt=True)[0]

            while not response:
                print("no response")
                time.sleep(10)
                response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                                      max_tokens=DEFAULT_MAX_TOKENS, for_creating_prompt=True)[0]

            prompt_template = response

            if is_prompt_valid(prompt_template, task_variables):

                prompt_template = prompt_template.replace("Prompt:", "").strip()
                prompt_template = correct_prompt(prompt_template, task_variables)

                if prompt_template not in prompt_templates_all:

                    if prompt_template.startswith('"') and prompt_template.endswith('"'):
                        prompt_template = prompt_template[1:-1]

                    prompt_templates_all.append(prompt_template)
                    cur_prompt_templates.append(prompt_template)

                    try:
                        df = insert_prompt_row(df=df,
                                               prompt_idx=cur_index,
                                               default_prompt=False,
                                               method="rephrase",
                                               task_description=None,
                                               task_description_length=None,
                                               task_description_source=None,
                                               language_style=None,
                                               meta_prompt=request,
                                               model_name=MODEL_NAME,
                                               model_temperature=cur_temperature,
                                               max_tokens=DEFAULT_MAX_TOKENS,
                                               model_response=response,
                                               prompt_template=prompt_template,
                                               prompt_format=get_prompt_format(prompt_template),
                                               prompt_length=get_text_num_tokens(prompt_template))
                    except:
                        print("error inserting row")
                        continue

                    print(cur_index)
                    cur_index += 1
                else:
                    print("prompt template already exists")
                    cur_temperature += 0.05
                    cur_temperature = min(cur_temperature, 1.0)
            else:
                print("prompt is not valid (missing keywords)")
                cur_temperature += 0.05
                cur_temperature = min(cur_temperature, 1.0)



    return df


def add_rephrase_prompts_bbh(df, task_information, meta_prompt_template, task_variables, num_of_templates_to_generate=1):
    print("Adding rephrase prompts...")
    cur_index = df["prompt idx"].max() + 1
    prompt_templates_all = []

    request = meta_prompt_template.substitute(task_information=task_information)

    cur_temperature = MODEL_TEMPERATURE
    while len(prompt_templates_all) < (num_of_templates_to_generate):

        response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                              max_tokens=DEFAULT_MAX_TOKENS, for_creating_prompt=True)[0]

        while not response:
            print("no response")
            time.sleep(10)
            response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                                  max_tokens=DEFAULT_MAX_TOKENS, for_creating_prompt=True)[0]

        prompt_template = response

        if is_prompt_valid(prompt_template, task_variables):

            prompt_template = prompt_template.replace("Prompt:", "").strip()
            prompt_template = correct_prompt(prompt_template, task_variables)

            if prompt_template not in prompt_templates_all:

                if prompt_template.startswith('"') and prompt_template.endswith('"'):
                    prompt_template = prompt_template[1:-1]

                prompt_templates_all.append(prompt_template)

                try:
                    df = insert_prompt_row(df=df,
                                           prompt_idx=cur_index,
                                           default_prompt=False,
                                           method="rephrase",
                                           task_description=None,
                                           task_description_length=None,
                                           task_description_source=None,
                                           language_style=None,
                                           meta_prompt=request,
                                           model_name=MODEL_NAME,
                                           model_temperature=cur_temperature,
                                           max_tokens=DEFAULT_MAX_TOKENS,
                                           model_response=response,
                                           prompt_template=prompt_template,
                                           prompt_format=get_prompt_format(prompt_template),
                                           prompt_length=get_text_num_tokens(prompt_template))
                except:
                    print("error inserting row")
                    continue

                print(cur_index)
                cur_index += 1
            else:
                print("prompt template already exists")
                cur_temperature += 0.05
                cur_temperature = min(cur_temperature, 1.0)
        else:
            print("prompt is not valid (missing keywords)")
            cur_temperature += 0.05
            cur_temperature = min(cur_temperature, 1.0)

    return df


def add_rephrase_prompts(df, default_prompt_templates, task_information, meta_prompt_template, task_variables, examples_with_explanations, num_of_templates_to_generate=1):
    if not examples_with_explanations:  # add rephrase prompts as we did specifcally for lmentry
        return add_rephrase_prompts_lmentry(df, default_prompt_templates, meta_prompt_template, task_variables, num_of_templates_to_generate)
    else:  # add repharse prompts as we did specifically for bbh:
        return add_rephrase_prompts_bbh(df, task_information, meta_prompt_template, task_variables, num_of_templates_to_generate)


def add_meta_cot_prompts(df, default_prompt_templates, hint, meta_prompt_template, task_variables, language_style=None, language_style_str=None, num_of_templates_to_generate=1):
    print("Adding meta cot prompts...")
    cur_index = df["prompt idx"].max() + 1
    cur_temperature = MODEL_TEMPERATURE

    if not language_style_str:
        language_style_str = ""

    prompt_templates = []
    task_explanations = []

    default_prompt_templates_str = '\n\n'.join(default_prompt_templates)

    request = meta_prompt_template.substitute(prompt_templates=default_prompt_templates_str,
                                              hint=hint,
                                              language_style=language_style_str)

    while len(prompt_templates) < num_of_templates_to_generate:

        if not language_style_str:
            request.replace("LANGUAGE_STYLE:\n\n\n", "")

        response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                              max_tokens=META_COT_MAX_TOKENS, for_creating_prompt=True)[0]

        while not response:
            time.sleep(10)
            response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                                  max_tokens=META_COT_MAX_TOKENS, for_creating_prompt=True)[0]
        try:
            response_dict = ast.literal_eval(response)
            if not isinstance(response_dict, dict):
                print("response in not a valid json")
                continue
        except:
            print("response in not a valid json")
            continue

        variants = response_dict["variants"]

        if not variants:
            print("variant list is empty..")
            while not response:
                time.sleep(10)
                response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                                      max_tokens=META_COT_MAX_TOKENS, for_creating_prompt=True,
                                      prev_response=prev_response, request_to_fix="Your output variant list is Empty.")[0]
            try:
                response_dict = ast.literal_eval(response)
                if not isinstance(response_dict, dict):
                    print("response in not a valid json")
                    continue
            except:
                print("response in not a valid json")
                continue
            continue

        variants = response_dict["variants"]

        variants_are_valid = True
        for variant in variants:
            if not is_prompt_valid(variant, task_variables):
                variants_are_valid = False
                break

        if not variants_are_valid:
            prev_response = response
            print("variants are not valid (missing keywords)")
            while not response:
                time.sleep(10)
                response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature,
                                      max_tokens=META_COT_MAX_TOKENS, for_creating_prompt=True,
                                      prev_response=prev_response, request_to_fix="Your output variants do not include all the keywords.")[0]
            try:
                response_dict = ast.literal_eval(response)
                if not isinstance(response_dict, dict):
                    print("response in not a valid json")
                    continue
            except:
                print("response in not a valid json")
                continue

        task_explanation = response_dict["task_explanation"]
        variants = response_dict["variants"]

        for variant in variants:
            if not is_prompt_valid(variant, task_variables):
                variants_are_valid = False
                break

        if not variants_are_valid:
            print("variants are still not valid (missing keywords)")
            continue

        if task_explanation not in task_explanations:
                task_explanations.append(task_explanation)
        else:
            print("task explanation already exists")
            # cur_temperature += 0.05
            # cur_temperature = min(cur_temperature, 1.0)
        for variant in variants:

            if variant in prompt_templates:
                print("prompt template already exists")
                cur_temperature += 0.05
                cur_temperature = min(cur_temperature, 1.0)
            if len(prompt_templates) < num_of_templates_to_generate:
                prompt_templates.append(variant)

                try:
                    df = insert_prompt_row(df=df,
                                           prompt_idx=cur_index,
                                           default_prompt=False,
                                           method="meta_cot",
                                           task_description=None,
                                           task_description_length=None,
                                           task_description_source=None,
                                           language_style=language_style,
                                           meta_prompt=request,
                                           model_name=MODEL_NAME,
                                           model_temperature=cur_temperature,
                                           max_tokens=META_COT_MAX_TOKENS,
                                           model_response=response,
                                           prompt_template=variant,
                                           prompt_format=get_prompt_format(variant),
                                           prompt_length=get_text_num_tokens(variant))
                except:
                    print("error inserting row")
                    continue
                print(cur_index)
                cur_index += 1

    return df, task_explanations


def get_task_descriptions(task_information, input_samples, get_description_template, num_of_task_descriptions=1):
    print("Get task descriptions...")
    cur_temperature = MODEL_TEMPERATURE

    task_descriptions = []

    request = get_description_template.substitute(task_information=task_information,
                                                  input1=input_samples[0],
                                                  input2=input_samples[1],
                                                  input3=input_samples[2])

    while request and len(task_descriptions) < num_of_task_descriptions:

        response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature, max_tokens=DEFAULT_MAX_TOKENS)[0]
        while not response:
            time.sleep(10)
            response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature, max_tokens=DEFAULT_MAX_TOKENS)[0]

        task_description = response.strip()
        if task_description in task_descriptions:  # if the task alreadt exists, increase the temperature to more variability in responses
            print("task description already exists")
            cur_temperature += 0.05
            cur_temperature = min(cur_temperature, 1.0)
        else:
            task_descriptions.append(task_description)

    return task_descriptions


def derive_general_prompt_from_meta_response(meta_prompt_response, prompt_variables, derive_prompt_template):

    request = derive_prompt_template + "Template variables: " + str(list(prompt_variables)) \
              + "\n\n" + meta_prompt_response + "\n\nPrompt template:"

    response = call_model(request=request, model_name=DERIVE_PROMPT_MODEL_NAME, echo=False, max_tokens=100)[0]
    while not response:
        time.sleep(10)
        response = call_model(request=request, model_name=DERIVE_PROMPT_MODEL_NAME, echo=False, max_tokens=100)[0]

    return response


def add_meta_desc_prompts(df, meta_desc_template, template_variables, derive_prompt_template, task_descriptions, descriptions_source, generated_templates_per_explanation=1):
    print("Adding meta desc prompts...")
    cur_index = df["prompt idx"].max() + 1
    cur_temperature = MODEL_TEMPERATURE

    prompt_templates = []

    for task_description in task_descriptions:
        request = meta_desc_template.substitute(task_description=task_description)

        for i in range(generated_templates_per_explanation):
            response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature, max_tokens=DEFAULT_MAX_TOKENS, for_creating_prompt=True)[0]
            while not response:
                time.sleep(10)
                response = call_model(request=request, model_name=MODEL_NAME, temperature=cur_temperature, max_tokens=DEFAULT_MAX_TOKENS, for_creating_prompt=True)[0]

            general_prompt = derive_general_prompt_from_meta_response(response, template_variables, derive_prompt_template)

            if general_prompt in prompt_templates:
                print("prompt template already exists")
                cur_temperature += 0.05
                cur_temperature = min(cur_temperature, 1.0)
                continue

            prompt_templates.append(general_prompt)
            try:
                df = insert_prompt_row(df=df,
                                       prompt_idx=cur_index,
                                       default_prompt=False,
                                       method="meta_desc",
                                       task_description=task_description,
                                       task_description_length=get_text_num_tokens(task_description),
                                       task_description_source=descriptions_source,
                                       language_style=None,
                                       meta_prompt=request,
                                       model_name=MODEL_NAME,
                                       model_temperature=MODEL_TEMPERATURE,
                                       max_tokens=DEFAULT_MAX_TOKENS,
                                       model_response=response,
                                       prompt_template=general_prompt,
                                       prompt_format=get_prompt_format(general_prompt),
                                       prompt_length=get_text_num_tokens(general_prompt))
            except:
                print("error inserting row")
                continue
            print(cur_index)
            cur_index += 1

    return df


def get_examples_with_explanations(proccessed_json_folder_path, task_name):
    with open(os.path.join(proccessed_json_folder_path, task_name + ".json"), 'r') as f:
        task_data = json.load(f)
    examples_with_explanations = task_data["more_info"]["examples_with_explanations"]
    return examples_with_explanations


def get_few_shot_derive_template(derive_prompt_template_given_few_shots_examples_folder_path,
                                 derive_prompt_template_given_few_shots_examples_file_names,
                                 task_information):

    derive_prompt_template = None

    if "general" in derive_prompt_template_given_few_shots_examples_file_names:
        with open(os.path.join(derive_prompt_template_given_few_shots_examples_folder_path,
                               derive_prompt_template_given_few_shots_examples_file_names["general"]),
                  'r', encoding='utf8') as f:
            derive_prompt_template = ''.join(f.readlines())
    else:
        if "multi_choice" in task_information:
            with open(os.path.join(derive_prompt_template_given_few_shots_examples_folder_path,
                                   derive_prompt_template_given_few_shots_examples_file_names["multi_choice"]),
                      'r', encoding='utf8') as f:
                derive_prompt_template = ''.join(f.readlines())
        elif "classification" in task_information:
            with open(os.path.join(derive_prompt_template_given_few_shots_examples_folder_path,
                                   derive_prompt_template_given_few_shots_examples_file_names["classification"]),
                      'r', encoding='utf8') as f:
                derive_prompt_template = ''.join(f.readlines())
        elif "seq2seq" in task_information:
            with open(os.path.join(derive_prompt_template_given_few_shots_examples_folder_path,
                                   derive_prompt_template_given_few_shots_examples_file_names["seq2seq"]),
                      'r', encoding='utf8') as f:
                derive_prompt_template = ''.join(f.readlines())

    return derive_prompt_template


def generate_prompts(prompts_outpath, task_info_folder_path, default_prompts_and_hints_json_path,
                     meta_prompts_module_path, task_names, num_calls_rephrase, num_calls_chain_of_thought,
                     num_calls_task_description_generation, num_calls_gradual_generation,
                     derive_prompt_template_given_few_shots_examples_folder_path,
                     derive_prompt_template_given_few_shots_examples_file_names,
                     examples_with_explanations):
    """
    This function automatically generates instruction templates for given tasks from a specific benchmark.

    :param prompts_outpath:
        The path where the generated prompts for the benchmark are saved.

    :param task_info_folder_path:
        The path to the folder containing information for all benchmark tasks.

    :param default_prompts_and_hints_json_path:
        The path to the JSON file containing the default prompts and hints for each task in the benchmark.

    :param meta_prompts_module_path:
        The path to the module containing meta prompts for the benchmark (without the .py extension).

    :param task_names:
        The names of the tasks in the benchmark for which prompts will be generated.

    :param num_calls_rephrase:
        The number of calls to the rephrasing generation method for each task (note: there are two rephrasing 
        meta-prompt templates, so 'x' calls will result in 2x prompts).

    :param num_calls_chain_of_thought:
        The number of calls to the chain-of-thought generation method for each task (note: this method results
        in a task description and several prompts per call).

    :param num_calls_task_description_generation:
        The number of calls to the task description generation method for each task (note: this is in addition
        to the task descriptions generated by the chain-of-thought method).

    :param num_calls_gradual_generation:
        The number of calls to the gradual generation method for each task (note: there are two gradual 
        generation meta-prompt template. Assuming there are 'm' task descriptions generated by the 
        chain-of-thought method and 'n' calls for task description generation, 'x' calls to this method will 
        result in 2x(m+n) prompts).

    :param derive_prompt_template_given_few_shots_examples_folder_path:
        The path to the folder containing the few-shot examples used to derive the final prompt templates in the 
        gradual generation method for each task.

    :param derive_prompt_template_given_few_shots_examples_file_names:
        The names of the files containing the few-shot examples used to derive the final prompt templates in the 
        gradual generation method for each task.

    :param examples_with_explanations:
        A boolean indicating whether the benchmark includes examples with explanations (e.g., bbh).

    :return:
        This function does not return a value, but it saves the generated prompts for each task at the path
        specified in 'prompts_outpath', with the filename pattern "task_name_prompts.xlsx".

    """

    imported_module = __import__(meta_prompts_module_path)

    if not os.path.exists(prompts_outpath):
        os.makedirs(prompts_outpath)

    for task_name in task_names:

        default_prompts, hint_str = get_default_prompts_and_hints(default_prompts_and_hints_json_path, task_name)
        task_information = hint_str.split("Keywords:")[0].strip() + "\nDefault prompt:\n" + default_prompts[0]

        if not examples_with_explanations:
            task_examples_with_explanations = None
        else:
            task_examples_with_explanations = get_examples_with_explanations(task_info_folder_path, task_name)

        task_variables = re.findall('{([^}]+)', default_prompts[0])

        xlsx_out_path = prompts_outpath + "/" + task_name + "_prompts.xlsx"

        df = pd.DataFrame()
        df = add_default_prompts(df, default_prompts)

        df = add_rephrase_prompts(df, default_prompts, task_information, imported_module.rephrase_template, task_variables, examples_with_explanations, num_of_templates_to_generate=num_calls_rephrase)
        df = add_rephrase_prompts(df, default_prompts, task_information, imported_module.rephrase_to_GPT3_template, task_variables, examples_with_explanations, num_of_templates_to_generate=num_calls_rephrase)

        df, task_descriptions_cot = add_meta_cot_prompts(df, default_prompts, hint_str, imported_module.meta_cot_template, task_variables, num_of_templates_to_generate=num_calls_chain_of_thought)

        input_samples = default_prompts
        if examples_with_explanations:
            input_samples = task_examples_with_explanations

        task_descriptions = get_task_descriptions(task_information, input_samples, imported_module.get_task_description_template, num_of_task_descriptions=num_calls_task_description_generation)

        if len(task_descriptions_cot) > num_calls_task_description_generation:
            task_descriptions_cot = task_descriptions_cot[:num_calls_task_description_generation]

        derive_prompt_template = get_few_shot_derive_template(derive_prompt_template_given_few_shots_examples_folder_path,
                                                              derive_prompt_template_given_few_shots_examples_file_names,
                                                              task_information)

        df = add_meta_desc_prompts(df, imported_module.meta_desc_1_template, task_variables, derive_prompt_template, task_descriptions, descriptions_source='get_desc', generated_templates_per_explanation=num_calls_gradual_generation)
        df = add_meta_desc_prompts(df, imported_module.meta_desc_1_template, task_variables, derive_prompt_template, task_descriptions_cot, descriptions_source='cot_desc', generated_templates_per_explanation=num_calls_gradual_generation)

        df = add_meta_desc_prompts(df, imported_module.meta_desc_2_template, task_variables, derive_prompt_template, task_descriptions, descriptions_source='get_desc', generated_templates_per_explanation=num_calls_gradual_generation)
        df = add_meta_desc_prompts(df, imported_module.meta_desc_2_template, task_variables, derive_prompt_template, task_descriptions_cot, descriptions_source='cot_desc', generated_templates_per_explanation=num_calls_gradual_generation)

        df.to_excel(xlsx_out_path, engine='xlsxwriter', index=False)


if __name__ == '__main__':

    # generate prompts for the LMentry benchmark:
    generate_prompts(prompts_outpath="generated_prompts/LMentry",
                     task_info_folder_path="benchmarks/LMentry/LMentry jsons processed",
                     default_prompts_and_hints_json_path="benchmarks/LMentry/default_prompts_and_hints.json",
                     meta_prompts_module_path="meta_prompt_templates_lmentry",
                     task_names=["word_not_containing", "ends_with_word", "word_before", "any_words_from_category",
                                 "rhyming_word", "less_letters", "all_words_from_category", "more_letters",
                                 "homophones", "first_alphabetically"],
                     num_calls_rephrase=1,
                     num_calls_chain_of_thought=1,
                     num_calls_task_description_generation=1,
                     num_calls_gradual_generation=1,
                     derive_prompt_template_given_few_shots_examples_folder_path="derive_prompt_template_few_shots",
                     derive_prompt_template_given_few_shots_examples_file_names={"general": "derive_prompt_template_few_shots_lmentry.txt"},  # we used the same few-shots examples for deriving instruction templates for all tasks in LMentry
                     examples_with_explanations=False)

    # generate prompts for the Big Bench Hard benchmark:
    generate_prompts(prompts_outpath="generated_prompts/big_bench_hard",
                     task_info_folder_path="benchmarks/big_bench_hard/big_bench_hard jsons processed",
                     default_prompts_and_hints_json_path="benchmarks/big_bench_hard/default_prompts_and_hints.json",
                     meta_prompts_module_path="meta_prompt_templates_bbh",
                     task_names=["causal_judgement", "disambiguation_qa", "dyck_languages", "formal_fallacies", "geometric_shapes",
                                 "hyperbaton", "logical_deduction_three_objects", "logical_deduction_five_objects",
                                 "logical_deduction_seven_objects", "movie_recommendation", "navigate", "object_counting",
                                 "penguins_in_a_table", "ruin_names", "salient_translation_error_detection", "snarks",
                                 "sports_understanding", "web_of_lies", "word_sorting"],
                     num_calls_rephrase=1,
                     num_calls_chain_of_thought=1,
                     num_calls_task_description_generation=1,
                     num_calls_gradual_generation=1,
                     derive_prompt_template_given_few_shots_examples_folder_path="derive_prompt_template_few_shots",
                     derive_prompt_template_given_few_shots_examples_file_names={"classification": "derive_prompt_template_few_shots_bbh_c.txt",
                                                                                 "multi_choice": "derive_prompt_template_few_shots_bbh_mc.txt",
                                                                                 "seq2seq": "derive_prompt_template_few_shots_bbh_s2s.txt"},
                     examples_with_explanations=True)
