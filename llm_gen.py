import json

from ceo import get_openai_model
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
model = get_openai_model(temp=0.5, top_p=0.8)

data = {
    'prompt': [],
    'human_result': [],
    'llm_result': []
}

with open('./script_review_alpaca_2025-02-28-15-52-27.json', 'r', encoding='utf-8') as f:
    data_json = json.load(f)

for i, item in enumerate(data_json):
    prompt = item['instruction'] + item['input']
    human_result = item['output']
    llm_result = model.invoke(prompt).content
    data['prompt'].append(prompt)
    data['human_result'].append(human_result)
    data['llm_result'].append(llm_result)
    print(f'Round {i+1}')
    print('Prompt:', prompt)
    print('Response:', llm_result, end='\n\n')
    df = pd.DataFrame(data)
    df.to_csv('./test_result/gpt4o-mini-retry.csv', index=False, encoding='utf-8')
