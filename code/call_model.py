import openai
from api_secrets import API_KEY
openai.api_key = API_KEY

import backoff

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError), max_time=60)  # this catches rate errors and server errors and retries in exponential time steps
def call_model(request, model_name, temperature=0.3, max_tokens=50, for_creating_prompt=False, echo=True,
               prev_response=None, request_to_fix=None):

    tokens = None
    logprobs = None

    try:
        if model_name == "cgpt":

            if for_creating_prompt:
                if not request_to_fix:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0301",
                        messages=[{"role": "system", "content": "You are a prompt creating assistant."},
                                  {"role": "user", "content": request}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0301",
                        messages=[{"role": "system", "content": "You are a prompt creating assistant."},
                                  {"role": "user", "content": request},
                                  {"role": "assistant", "content": prev_response},
                                  {"role": "user", "content": request_to_fix}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            else:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-0301",
                    messages=[{"role": "user", "content": request}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            response = completion["choices"][0]["message"]["content"]

        else:

            parameters = {'max_tokens': max_tokens,
                          'top_p': 0,
                          'temperature': temperature,
                          'stop': ["#end#"],
                          'logprobs': 5,
                          'engine': model_name,
                          'prompt': request,
                          'echo': echo}

            raw_response = openai.Completion.create(**parameters)
            response = raw_response['choices'][0]['text']
            tokens = raw_response['choices'][0]['logprobs']['tokens']
            logprobs = raw_response['choices'][0]['logprobs']['token_logprobs'][:-1]

    except (openai.error.RateLimitError, openai.error.APIError) as e:
        print("Exception occurred: ", str(e))
        return None, None, None

    return response, tokens, logprobs


