#!/usr/bin/python

import sqlalchemy
import json
import boto3
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import tiktoken
import sys
import io
from pathlib import Path

import logging

from .utils import run_dfs

logging.getLogger().setLevel('INFO')

enc = tiktoken.encoding_for_model("gpt-4")
bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2')

def get_claude_completion(prompt, temperature=0, max_new_tokens=10000):
    """
    `prompt`: The input to LLM
    """
    if not prompt or len(prompt) == 0:
        return None
    try:
        body = json.dumps({
            "prompt": f"\n\nHuman: {prompt} \n\nAssistant:",
            "max_tokens_to_sample": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        })
        modelId = 'anthropic.claude-v2'
        #modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
        accept = 'application/json'
        contentType = 'application/json'

        response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

        response_body = json.loads(response.get('body').read())
        # text
        # print(response_body.get('completion'))
        return response_body.get('completion')
    except Exception as e:
        logging.error(f'{type(e)}: {e}')
        return None

def get_prompt(column_name, table_name, value_counts, with_reason):
    """
    Generate the prompt for detecting the data type of `column_name` in `table_name`.
    """
    LIMIT = 800
    str_list = []
    total_cnt = 0
    for k,v in value_counts.items():
        k_str = str(k[0]) if isinstance(k, tuple) else str(k)
        str_len = len(enc.encode(k_str))
        total_cnt += str_len
        if total_cnt > LIMIT:
            break
        str_list.append(k_str)
    prompt = f"Determine the most suitable data type for the column `{column_name}` in the table `{table_name}`. Consider the following data type options: categorical, ID, numerical, natural language text, datetime, or other. Example values from the column are:\n" + "\n".join(str_list) + "."
    if with_reason:
        prompt += """ Explain your choice with a justification. Format your response as JSON: \n:```{"column_type": "Categorical|ID|Numerical|Text|DateTime|Other", "reason": "Your explanation here"}``` ."""
    else:
        prompt += """ Please format your choice without explanation as JSON: \n:```{"column_type": "Categorical|ID|Numerical|Text|DateTime|Other"}``` ."""
    return prompt


def get_time_column_prompt(table_name, column_type_dict, data):
    """
    Generate the prompt for detecting which column is the time column for DFS, given the data type of each column.
    """
    prompt = f"Determine which column represents the time for each row in the table `{table_name}`.  The data type for each column is:\n"
    for column, dtype in column_type_dict.items():
        prompt += f"{column}: {dtype}\n"
    prompt += f"A sample data of 50 rows in CSV format goes as follows:\n"
    buf = io.StringIO()
    data.sample(50).to_csv(buf, index=False)
    prompt += buf.getvalue()
    prompt += "\nPlease directly give the column name, without any explanation."
    return prompt


def eval_reason(text:str):
    text = text.strip(" `")
    text = text.replace("json", '')
    try:
        return json.loads(text)
    except:
        return {}

def eval_result(text:str):
    json_str = text.strip().strip(' ```json\n').strip('\n```')
    json_str = json_str.split('```')[0]
    data = json.loads(json_str)
    column_type = data['column_type']
    try:
        return column_type
    except:
        return {}
    
def judge_column_type(data, path, with_reason=False, row_limit=1000):
    """
    Return a dict {column_name: dtype} where dtype can be either of the following:

    * Categorical
    * Numerical
    * Text
    * ID (which will be treated as key column for DFS)
    * DateTime
    * Other
    * null (which will be ignored later)
    """
    column_type = {}

    for column in data.columns:
        logging.info(f'Identifying column data type for {column}...')
        column_ = data[column].iloc[:1000]
        value_counts = column_.value_counts()
        if len(value_counts) <= 1:
            column_type[column] = 'null'  # no value or single value: this column can be dropped
            continue
        prompt = get_prompt(column, path, value_counts, with_reason)
        logging.debug(prompt)
        claude_answer = get_claude_completion(prompt)
        if with_reason:
            claude_answer = eval_reason(claude_answer)
        else:
            claude_answer = eval_result(claude_answer)
            if claude_answer not in ['Categorical', 'Numerical', 'Text', 'Other', 'ID', 'DateTime']:
                print('**claude answer error.')
                claude_answer = 'null'
        logging.debug(claude_answer)
        column_type[column] = claude_answer
    return column_type


def find_time_column(data, name, column_type_dict):
    """
    Return the name of the timestamp column of the table.
    """
    for trial in range(5):
        prompt = get_time_column_prompt(name, column_type_dict, data)
        claude_answer = get_claude_completion(prompt)
        logging.debug(claude_answer)
        column_name = claude_answer.strip()

        if column_name in data.columns:
            return column_name
    else:
        raise Exception('Time column prompting failed 5 times; consider changing prompt.')


def dfs_on_table(
    input_df: pd.DataFrame,
    df_name: str,
    target_column: str,
    depth: int
):
    """
    The main entrance that runs DFS with given depth, the table, the name of
    the table, and the target column.
    """
    column_type_dict = judge_column_type(input_df, df_name)
    time_column = find_time_column(input_df, df_name, column_type_dict)
    logging.info(f'Column type dict: {column_type_dict}')
    logging.info(f'Timestamp column: {time_column}')
    output_df = run_dfs(input_df, df_name, target_column, time_column, column_type_dict, depth)
    assert target_column in output_df.columns, "Target column is not included in output."
    return output_df


if __name__ == '__main__':
    path = sys.argv[1]
    target_column = sys.argv[2]
    depth = int(sys.argv[3])
    output_path = sys.argv[4]

    input_df = pd.read_parquet(path)
    df_name = Path(path).stem
    output_df = dfs_on_table(input_df, df_name, target_column, depth)
    logging.info(f'Writing to {output_path}...')
    output_df.to_parquet(output_path)
