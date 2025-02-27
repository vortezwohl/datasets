import pandas as pd
import json

STANDARD = '''
<评估标准>
1. 主线 (核心故事线): 评估主线故事的质量和潜力。
    - 通过标准: 主线 **描述清晰且集中**, 核心矛盾突出且简单易懂, 但矛盾不宜复杂；剧情 **节奏紧凑且脉络简单**，符合短剧的格式要求；逻辑 **自洽**；主线具备支撑剧情发展和吸引观众的基础，
    - 不通过标准: 主线 **矛盾分散**，缺乏明确的冲突焦点；或者剧情 **节奏拖沓或过于复杂**，不符合短剧的紧凑要求；或者 **前后逻辑不自洽**，难以支撑剧情发展。
2. 人设 (角色设定): 评估角色的吸引力和合理性。
    - 通过标准: 角色 **性格鲜明**，具备一定的辨识度；角色行为 **符合逻辑**，且有一定背景支撑；主角 **目的与行动一致**，动机明确；
    - 不通过标准: 角色 **扁平化**，缺乏鲜明个性和记忆点；或者 角色行为 **不符合逻辑**；或者 主角 **目的与行动不一致**；或者存在 **人设矛盾** 的情况。
3. 钩子 (关键悬念点): 评估剧情钩子的质量和吸引力。
    - 通过标准: **开篇或关键情节设置了一定的冲突或信息差**，初步吸引眼球；剧情 **情绪表达基本到位**，能初步调动观众情绪；**卡点设置合理**，能引发观众一定的期待。
    - 不通过标准: **开篇或关键情节缺乏强冲突或信息差**，平淡无奇；或者 剧情 **情绪不足**；或者 **卡点设置无效**，无法引发观众期待。
4. 投流 (商业化潜力): 评估大纲的商业变现潜力。
    - 通过标准: 剧情 **吸睛事件较密集**，可提炼出一定数量的投流素材。
    - 不通过标准: 剧情 **缺乏吸睛事件**，难以提炼投流素材；或者 **素材质量低**，不符合目标市场偏好**。
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
        instruction = ('<任务>依据给定评估标准,对用户提供的短剧大纲进行评估，并给出评估结果和改进建议</任务>'
                       '<目标>按规定格式给出评估结果和改进建议</目标>')
        instruction += STANDARD
        _input = row['outline']

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
    return alpaca_data


def convert_to_dpo_format(df):
    dpo_data = []
    for _, row in df.iterrows():
        # 提取 outline 作为 instruction
        instruction = ('<任务>依据给定评估标准,对用户提供的短剧大纲进行评估，并给出评估结果和改进建议</任务>'
                       '<目标>按规定格式给出评估结果和改进建议</目标>')
        instruction += STANDARD
        _input = row['outline']

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
    return dpo_data


def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')


def main():
    from pprint import pp
    # 文件路径
    csv_file_path = 'raw_data/llm_dataset_post.csv'  # 原始 CSV 文件
    alpaca_jsonl_file_path = 'script_review_alpaca.jsonl'  # 输出 JSONL 文件
    dpo_jsonl_file_path = 'script_review_dpo.jsonl'  # 输出 JSONL 文件
    # 读取数据
    df = read_csv_data(csv_file_path)
    # 转换数据
    alpaca_data = convert_to_alpaca_format(df)
    dpo_data = convert_to_dpo_format(df)
    # 保存为 jsonl 文件
    save_to_jsonl(alpaca_data, alpaca_jsonl_file_path)
    save_to_jsonl(dpo_data, dpo_jsonl_file_path)


if __name__ == "__main__":
    main()
