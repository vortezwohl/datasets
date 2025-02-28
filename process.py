from datetime import datetime
from pprint import pp

import pandas as pd
import json


STANDARD = '''
<评估标准>
1. 主线 (核心故事线): 
    评估主线故事是否有充足的看点
    - 评估标准
        1. **主线剧情清晰**, 核心矛盾冲突简单易懂, 让观众无需理解能力即可看懂, 主要矛盾需集中于一处(或一人/一团体/一实体)
        2. **剧情节奏紧凑且脉络简单**, 不宜过于悬疑, 符合短剧快节奏-通俗-简单的基本要求, 剧情事件一环扣一环
        3. **逻辑自洽**, 剧情前后呼应, 人物设定符合剧情背景且无违和感
        4. (非必须) **看点丰富**, 剧情要有丰富且精彩不断的看点, 以长期不断观众观看
2. 人设 (角色设定): 
    评估角色的人格吸引力和以及角色能否合理地融入剧情
    - 评估标准 (基本满足即可)
        1. **角色性格鲜明**, 具备很好的辨识度, 不应该存在扁平化或缺乏个性的角色
        2. **角色动机明确**, 角色有明确的动机, 且所有行为与动作都建立在其基本动机之上, 且逻辑自洽
        在人设评估阶段暂不考虑人设之间的关系, 主要考虑人设和剧情大背景之间的关联程度
3. 钩子 (关键悬念点): 
    评估剧情钩子能否引导观众继续观看剧集
    - 评估标准 (满足其一就算通过)
        1. **在开篇或关键情节设置了强烈冲突**, 吸引观众眼球
        2. **在开篇或关键情节刻意制造信息差**, 引发观众好奇心理
        3. **将剧情情绪烘托到高位, 戛然而止**, 令观众深入沉浸, 渴望继续观看
        4. **剧情卡点设置恰当, 破坏关键剧情的连贯性**, 引发观众继续观看的强烈期待
4. 投流 (商业化潜力): 
    评估大纲的推广传播潜力以及商业变现潜力
    - 评估标准 (满足其一就算通过)
        1. **吸睛事件**，可提炼出一定数量的投流素材
        2. **矛盾冲突激烈**, 牵动观众情绪
</评估标准>
<输出格式>
```json
[
    {"dimension": "主线", "analysis": ..., "result": "通过/不通过"},
    {"dimension": "人设", "analysis": ..., "result": "通过/不通过"},
    {"dimension": "钩子", "analysis": ..., "result": "通过/不通过"},
    {"dimension": "投流", "analysis": ..., "result": "通过/不通过"}
]
```
</输出格式>'''.replace(' ', '')


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
        return _data
    except json.JSONDecodeError as e:
        print(f'Failed to parse: \n{_raw_data}\nError: {e}')
        return None


def read_csv_data(file_path):
    return pd.read_csv(file_path)


def convert_to_alpaca_format(df):
    alpaca_data = []
    for _, row in df.iterrows():
        # 提取 outline 作为 instruction
        instruction = ('<任务>依据给定评估标准, 对用户提供的短剧大纲进行多维度评估, 给出各个维度的评估结果(通过/不通过), 对于不通过的维度, 请给出其原因和改进建议</任务>'
                       '<目标>按规定格式给出评估结果和改进建议</目标>')
        instruction += STANDARD
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
            "instruction": instruction,
            "input": _input,
            "output": output_data,
            "system": ""
        }
        alpaca_data.append(alpaca_entry)
    print('Alpaca:')
    pp(alpaca_data)
    return alpaca_data


def convert_to_dpo_format(df):
    dpo_data = []
    for _, row in df.iterrows():
        # 提取 outline 作为 instruction
        instruction = ('<任务>依据给定评估标准, 对用户提供的短剧大纲进行多维度评估, 给出各个维度的评估结果(通过/不通过), 对于不通过的维度, 请给出其原因和改进建议</任务>'
                       '<目标>按规定格式给出评估结果和改进建议</目标>')
        instruction += STANDARD
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
        json.dump(data, f, ensure_ascii=False)


def main(get_index: int = -1):
    from pprint import pp
    # 文件路径
    now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    csv_file_path = 'raw_data/llm_dataset_post.csv'  # 原始 CSV 文件
    alpaca_jsonl_file_path = f'script_review_alpaca_{now}.json'  # 输出 JSON 文件
    dpo_jsonl_file_path = f'script_review_dpo_{now}.json'  # 输出 JSON 文件
    # 读取数据
    df = read_csv_data(csv_file_path)
    # 转换数据
    alpaca_data = convert_to_alpaca_format(df)
    dpo_data = convert_to_dpo_format(df)

    # 保存为 json 文件
    if get_index < 0:
        save_to_json_list(alpaca_data, alpaca_jsonl_file_path)
        save_to_json_list(dpo_data, dpo_jsonl_file_path)
    else:
        print('Instruction Data Example:')
        print(alpaca_data[get_index]['instruction'])
        print(alpaca_data[get_index]['input'])
        print('Label Data Example:')
        pp(alpaca_data[get_index]['output'])


if __name__ == "__main__":
    main()
