# Checking hallucinations using Google FACTS paper's prompt

# %%
from typing import Literal, Callable
from tqdm import tqdm
import pandas as pd

# %%
# prompts used by Google FACTS paper 
facts_judge_prompt  = """
Your task is to check if the Response is accurate to the Evidence.
Generate 'Accurate' if the Response is accurate when verified according to the Evidence,
or 'Inaccurate' if the Response is inaccurate (contradicts the evidence) or cannot be
verified.
**Evidence**\n\n{context}\n\n**End of Evidence**\n
**Response**:\n\n{response}\n\n**End of Response**\n
Let's think step-by-step.
"""

# %%
import dotenv

from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

genai.configure(api_key=dotenv.get_key(dotenv_path=".env", key_to_get='GEMINI_API_KEY'))

dotenv.load_dotenv()
openai_client = OpenAI()
anthropic_client = Anthropic()

# %%

def gen_response_by_openai(prompt:str, model: str) -> str:
    client = openai_client # global 
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def gen_response_by_anthropic(prompt:str, model: str) -> str:
    client = anthropic_client # global
    completion = anthropic_client.messages.create(
        model=model,
        system = "You are a helpful assistant.",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    return completion.content[0].text

def gen_response_by_gemini(prompt:str, model: str) -> str:
    model = genai.GenerativeModel(model)
    response = model.generate_content(prompt)
    return response.text

def judge_by_llm(context:str, response:str, model: str) -> Literal[0,1]:
    prompt = facts_judge_prompt.format(context=context, response=response)

    if "gpt" in model:
        response = gen_response_by_openai(prompt, model)
    elif "claude" in model:
        response = gen_response_by_anthropic(prompt, model)
    elif "gemini" in model:
        response = gen_response_by_gemini(prompt, model)

    response = response.lower()

    # find the word "accurate" or "inaccurate" near the end of the response

    window = min(12, len(response))
    while window <= len(response): 
        # print ("response[-window]: ", response[-window:])
        if "inaccurate" in response[-window:]:
            return 0
        elif "accurate" in response[-window:]: # must have a space before accurate
            return 1
        window += 10

    return -1 # error 

# judge_by_llm("The sky is blue", "The sky is blue", "claude-3-5-sonnet-latest")
# judge_by_llm("The sky is blue", "The sky is blue", "gemini-1.5-pro")

# %%
# iterate over the rows of the dataframe and judge each response
# save the DF with the new column "judgement" in every iteration 

def loop_over_df(deepseek_version: str, model: str):    
    df = pd.read_csv(f'{deepseek_version}.csv')

    judgement_column = f'{model}_judgement'
    df[judgement_column] = -1

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if row[judgement_column] != -1: # already judged,
            continue 
        context = row['source']
        response = row['summary']
        judgement = judge_by_llm(context, response, model)
        df.at[i, judgement_column] = judgement

        df.to_csv(f'{deepseek_version}_{model}_judgement.csv', index=False) # save after every iteration


# %%

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Judge the responses in the R1 dataset')
    parser.add_argument('--model', type=str, help='The model to use for judging the responses')
    parser.add_argument('--version', type=str, help='The version of the DeepSeek model')
    args = parser.parse_args()

    loop_over_df(args.version, args.model)

