from datetime import datetime
from pprint import pp

import pandas as pd
import json

INSTRUCTION = ('# **任务描述:**\n'
               '你是一个短剧评估专家，请根据以下步骤对短剧大纲的质量进行评估，并输出评估结果。')

STEP = '''
# **评估步骤:**
1. **理解内容**: 阅读短剧大纲，提取主线、人设、钩子和投流内容。
2. **评估主线**: 分析主线内容，包括设定、看点、主题、角色前后行为。
   - 不通过的标准（严格）:
     - 设定 >= 5个，剧情太复杂。（设定是剧情刚开始的一些特殊设定，例如植物人、穿越、重生等，不考虑剧情发展过程中的情节）
     - 看点 <= 3个，剧情没看点。
     - 主题是纯虐或亲情，主线内容不吸引海外用户
     - 角色前后行为不合逻辑
3. **评估人设**:
   - 不通过的标准（宽松）:
     - 人设和主线矛盾
     - 人设没有推动剧情
4. **评估钩子**:
   - 不通过的标准: 钩子类型不属于以下之一（严格）:
     - 英雄救美
     - 亲子相认
     - 逆袭打脸（包括主角高姿态出场、实力曝光、容貌或身材变美等）
     - 女主患绝症或患病被曝光
     - 真相曝光
     - 身份曝光（也叫掉马甲）
     - 吸睛画面（色情、赤裸等）
5. **评估投流**:
   - 不通过的标准: 投流内容平淡，缺少尖锐的冲突。（严格）
'''

FORMAT = '''
# **输出格式:**
输出一个键名为 "result" 的 JSON 列表, 该列表包含四个 JSON 对象, 分别对应四个评估维度（主线、人设、钩子和投流）, 每个 JSON 对象包含以下键:
- "dimension": str，评估维度名称, 取值 ∈ ["主线", "人设", "钩子", "投流"].
- "description": str，该维度的评估理由, 若评估为 "不通过", 必须详细说明具体原因; 若评估结果为 "通过", 简要说明优点即可.
- "result": str，该维度的评估结果 ("通过"/"不通过").
# **示例输出:**
```json
{
  "result": [
    {
      "dimension": "主线",
      "description": "主线设定过多，超过了5个，导致故事焦点不明确。",
      "result": "不通过"
    },
    {
      "dimension": "人设",
      "description": "人设与主线一致，且有效推动了剧情发展。",
      "result": "通过"
    },
    {
      "dimension": "钩子",
      "description": "钩子类型不符合要求，缺乏吸引观众的亮点。",
      "result": "不通过"
    },
    {
      "dimension": "投流",
      "description": "投流内容矛盾冲突明显，能够吸引观众。",
      "result": "通过"
    }
  ]
}
```
'''


def data_reform(_raw_data, llm: bool = False):
    try:
        _data = json.loads(_raw_data)
        if llm:
            _data = _data['result']
        for d in _data:
            res = d['result']
            del d['result']
            d['analysis'] = d['description']
            d['result'] = res
            del d['description']
        return json.dumps(_data, ensure_ascii=False)
    except json.JSONDecodeError as e:
        print(f'Failed to parse: \n{_raw_data}\nError: {e}')
        return None


def read_csv_data(file_path):
    return pd.read_csv(file_path, encoding='gbk')


def convert_to_openai_format(df):
    openai_data = []
    for _, row in df.iterrows():
        # 提取 outline 作为 instruction
        instruction = INSTRUCTION + STEP + FORMAT
        _input = '<短剧大纲>' + row['outline'] + '</短剧大纲>'

        # 提取 human_result_data 或 llm_result_data 作为 output（根据需求选择）
        # 这里选择 human_result_data 作为示例
        # 如果需要使用 llm_result_data，请修改以下行
        raw_output_data = str(row['human_result_data']).replace('\'', '\"')
        if raw_output_data is None or raw_output_data == 'nan':
            continue
        output_data = data_reform(raw_output_data)
        # 转换为 Alpaca 格式
        alpaca_entry = {
            "messages":[
                {
                    'role': 'system',
                    'content': instruction
                },
                {
                    'role': 'user',
                    'content': _input
                },
                {
                    'role': 'assistant',
                    'content': output_data
                }
            ]
        }
        openai_data.append(alpaca_entry)
    print('Openai:')
    pp(openai_data)
    return openai_data


def convert_to_alpaca_format(df):
    alpaca_data = []
    for _, row in df.iterrows():
        # 提取 outline 作为 instruction
        instruction = INSTRUCTION + STEP + FORMAT
        _input = '<短剧大纲>' + str(row['outline']) + '</短剧大纲>'

        # 提取 human_result_data 或 llm_result_data 作为 output（根据需求选择）
        # 这里选择 human_result_data 作为示例
        # 如果需要使用 llm_result_data，请修改以下行
        raw_output_data = str(row['human_result_data']).replace('\'', '\"')
        if raw_output_data is None or raw_output_data == 'nan':
            continue
        output_data = data_reform(raw_output_data)
        # 转换为 Alpaca 格式
        alpaca_entry = {
            "instruction": instruction,
            "input": _input,
            "output": output_data
        }
        alpaca_data.append(alpaca_entry)
    print('Alpaca:')
    pp(alpaca_data)
    return alpaca_data


def convert_to_dpo_format(df):
    dpo_data = []
    for _, row in df.iterrows():
        # 提取 outline 作为 instruction
        instruction = INSTRUCTION + STEP + FORMAT
        _input = '<短剧大纲>' + row['outline'] + '</短剧大纲>'

        # 提取 human_result_data 或 llm_result_data 作为 output（根据需求选择）
        # 这里选择 human_result_data 作为示例
        # 如果需要使用 llm_result_data，请修改以下行

        output_data_1 = str(row['human_result_data']).replace('\'', '\"')
        output_data_2 = str(row['llm_result_data']).replace('\'', '\"')
        if output_data_1 is None or output_data_2 is None or output_data_1 == 'nan' or output_data_2 == 'nan':
            continue
        output_data_1 = data_reform(output_data_1)
        output_data_2 = data_reform(output_data_2, True)
        # 转换为 dpo 格式
        dpo_entry = {
            "instruction": instruction,
            "input": _input,
            "chosen": output_data_1,
            "rejected": output_data_2,
            "system": ""
        }
        dpo_data.append(dpo_entry)
    print('DPO:')
    pp(dpo_data)
    return dpo_data


def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def save_to_json_list(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))


def main(get_index: int = -1):
    from pprint import pp
    # 文件路径
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    csv_file_path = 'raw_data/llm_dataset_pre_test.csv'  # 原始 CSV 文件
    alpaca_jsonl_file_path = f'script_review_alpaca_test_{now}.json'  # 输出 JSON 文件
    openai_jsonl_file_path = f'script_review_openai_{now}.jsonl'  # 输出 JSON 文件
    dpo_jsonl_file_path = f'script_review_dpo_{now}.json'  # 输出 JSON 文件
    # 读取数据
    df = read_csv_data(csv_file_path)
    # 转换数据
    alpaca_data = convert_to_alpaca_format(df)
    # dpo_data = convert_to_dpo_format(df)
    # openai_data = convert_to_openai_format(df)

    # 保存为 json 文件
    if get_index < 0:
        save_to_json_list(alpaca_data, alpaca_jsonl_file_path)
        # save_to_json_list(dpo_data, dpo_jsonl_file_path)
        # save_to_jsonl(openai_data, openai_jsonl_file_path)
    else:
        print('Instruction Data Example:')
        print(alpaca_data[get_index]['instruction'])
        print(alpaca_data[get_index]['input'])
        print('Label Data Example:')
        pp(alpaca_data[get_index]['output'])


if __name__ == "__main__":
    main()
