import json
import re

def transform_tool_format_gpt(tool):
    transformed_tool = {
        "type": "function",
        "function": {
            "name": tool['name'],
            "description": tool['summary'],
            "parameters":{
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
    for param in tool['parameters']:
        if param['type'] == 'array':
            if 'array of' in param['description']:
                start = param['description'].index('array of')
                item_des = param['description'][start + len("array of"):].strip()
            elif 'list of' in param['description']:
                start = param['description'].index('list of')
                item_des = param['description'][start + len("list of"):].strip()
            else:
                item_des = param['description']
            
            transformed_tool['function']['parameters']['properties'][param['name']] = {
                    "type": 'array',
                    "items": {
                            "type": "object",
                            "properties": {
                                param['name']: { 
                                    "type": "string", "description": item_des},
                            }
                        }
                }
        else:
            transformed_tool['function']['parameters']['properties'][param['name']] = {
                "type": param['type'],
                "description": param['description'],
            }
        
        if param.get('required'):
            transformed_tool['function']['parameters']['required'].append(param['name'])
    return transformed_tool


def get_tool_dict(gpt_format=False):
    tool_dict = {}
    with open('./generate_data/dataset_utils/tools.json', 'r') as f:
        toolkits = json.load(f)
        for toolkit in toolkits:
            kit_name = toolkit['toolkit']
            for tool in toolkit['tools']:
                tool_name = kit_name + tool['name']
                tool['name'] = tool_name
                if gpt_format:
                    tool_dict[tool['name']] = transform_tool_format_gpt(tool)
                else:
                    tool_dict[tool['name']] = tool
    return tool_dict
