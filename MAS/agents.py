import os
import asyncio
import numpy as np
import re
from typing import Dict, Literal, Optional, Tuple

from utils.llm_client import LLMClientFactory
from .agent_prompts import *


_factory = LLMClientFactory()
_sync_clients: Dict[Tuple[str, str, str], object] = {}
_async_clients: Dict[Tuple[str, str, str], object] = {}


def _get_sync_client(
    client_name: str = 'agent_sync',
    base_url_env: str = 'BASE_URL',
    api_key_env: str = 'OPENAI_API_KEY',
):
    cache_key = (client_name, base_url_env, api_key_env)
    if cache_key not in _sync_clients:
        client, _ = _factory.create_sync_client(
            client_name=client_name,
            model=None,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
        )
        _sync_clients[cache_key] = client
    return _sync_clients[cache_key]


def _get_async_client(
    client_name: str = 'agent_async',
    base_url_env: str = 'BASE_URL',
    api_key_env: str = 'OPENAI_API_KEY',
):
    cache_key = (client_name, base_url_env, api_key_env)
    if cache_key not in _async_clients:
        client, _ = _factory.create_async_client(
            client_name=client_name,
            model=None,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
        )
        _async_clients[cache_key] = client
    return _async_clients[cache_key]


def llm_invoke(prompt, model_type: str):
    client = _get_sync_client()
    response = client.chat.completions.create(
        model=model_type,
        messages=prompt,
        temperature=0,
        max_tokens=1024,
    ).choices[0].message.content

    return response


async def allm_invoke(prompt, model_type: str):
    aclient = _get_async_client()

    response = await aclient.chat.completions.create(
        model=model_type,
        messages=prompt,
        temperature=0,
        max_tokens=1024,
    )

    return response.choices[0].message.content


class Agent: 
    def __init__(self, system_prompt, model_type, attack_mode: str = "normal"): 
        self.model_type = model_type
        self.system_prompt = system_prompt 
        self.memory = []
        self.memory.append({"role": "system", "content": system_prompt})
        self.role = "normal"
        self.attack_mode = attack_mode

    def parser(self, response):
        if self.attack_mode == "TA":
            self.last_response = response
        else:
            splits = re.split(r'<[A-Z_ ]+>: ', str(response).strip())
            splits = [s for s in splits if s]
            if len(splits) == 2:
                answer = splits[-1].strip()
                reason = splits[-2].strip()
                self.last_response = {"answer": answer, "reason": reason}
            else:
                self.last_response = {"answer": None, "reason": response}
 
    def chat(self, prompt): 
        user_msg = {"role": "user", "content": prompt}
        self.memory.append(user_msg)
        response = llm_invoke(self.memory, self.model_type)
        self.parser(response)
        ai_msg = {"role": "assistant", "content": response}
        self.memory.append(ai_msg)
        
        return response
    
    def set_role(self, role: Literal["normal", "attacker", "infected"]): 
        self.role = role
    
    def get_role(self):
        return self.role
    
    def set_attack_mode(self, attack_mode: str):
        self.attack_mode = attack_mode
    
    def get_attack_mode(self):
        return self.attack_mode
    
    async def achat(self, prompt): 
        user_msg = {"role": "user", "content": prompt}
        self.memory.append(user_msg)
        response = await allm_invoke(self.memory, self.model_type)
        self.parser(response)
        ai_msg = {"role": "assistant", "content": response}
        self.memory.append(ai_msg)
        
        return response
    
class AgentGraph:
    def __init__(self, adj_matrix, system_prompts, attacker_idxes, model_type="gpt-4o-mini", attack_mode: str = "MA", dataset='gsm8k'): 
        self.num_agents = len(system_prompts)
        self.adj_matrix = adj_matrix
        self.system_prompts = system_prompts
        self.attacker_idxes = attacker_idxes
        self.model_type = model_type
        self.attack_mode = attack_mode
        self.dataset = dataset
        self.agents = []
        for i in range(len(adj_matrix)): 
            self.agents.append(Agent(system_prompts[i], model_type, attack_mode))
    
    def _get_ma_first_prompt(self, idx, query, context, ):
        if idx not in self.attacker_idxes: 
            prompt = MA_USER_PROMPT_FIRST.format(query=query)
        else: 
            prompt = MA_ATTACKER_SYS_PROMPT_FIRST.format(query=query, context=context)
        return prompt
    
    def _get_pi_first_prompt(self, idx, qa_data, ):
        if self.dataset == 'gsm8k':
            if idx not in self.attacker_idxes: 
                prompt = PI_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0])
            else:                 
                prompt = PI_ATTACKER_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0], qa_data_2=qa_data[2])
        else:
            if idx not in self.attacker_idxes: 
                prompt = PI_USER_PROMPT_FIRST.format(qa_data=qa_data[0])
            else: 
                prompt = PI_ATTACKER_USER_PROMPT_FIRST.format(qa_data=qa_data[0], qa_data_2=qa_data[2])
        return prompt
    
    def _get_ta_first_prompt(self, idx, user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad, ):
        if idx not in self.attacker_idxes: 
            prompt = TA_USER_PROMPT.format(available_tool_descriptions = available_tool_descriptions, input = user_instruction, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
        else: 
            prompt = TA_ATTACKER_USER_PROMPT.format(input_text=agent_scratchpad)
        return prompt
    
    def first_generate_agent(self, idx, *args, ):
        if self.attack_mode == "MA":
            query, context = args
            prompt = self._get_ma_first_prompt(idx, query, context)
        elif self.attack_mode == "PI":
            qa_data = args[0]
            prompt = self._get_pi_first_prompt(idx, qa_data)
        elif self.attack_mode == "TA":
            user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad = args[0]
            prompt = self._get_ta_first_prompt(idx, user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
        
        response = self.agents[idx].chat(prompt)
        return idx, response
    
    def first_generate(self, *args, ):
        agent_responses = []
        for idx in range(self.num_agents): 
            response = self.first_generate_agent(idx, *args, )
            agent_responses.append(response)
        
        return agent_responses
    
    async def afirst_generate_agent(self, idx, *args, ):
        if self.attack_mode == "MA":
            query, context = args[0]
            prompt = self._get_ma_first_prompt(idx, query, context)
        elif self.attack_mode == "PI":
            qa_data = args[0]
            prompt = self._get_pi_first_prompt(idx, qa_data)
        elif self.attack_mode == "TA":
            user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad = args[0]
            prompt = self._get_ta_first_prompt(idx, user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
            
        response = await self.agents[idx].achat(prompt)
        return idx, response
    
    async def afirst_generate(self, *args, ): 
        tasks = []
        for idx in range(self.num_agents): 
            tasks.append(asyncio.create_task(self.afirst_generate_agent(idx, *args, )))
        agent_responses = await asyncio.gather(*tasks)
        return agent_responses

    def _get_ma_regenerate_prompt(self, idx, prompt):
        if idx not in self.attacker_idxes: 
            prompt += MA_USER_PROMPT_RE
        else: 
            prompt += MA_ATTACKER_USER_PROMPT_RE
        return prompt
    
    def _get_pi_regenerate_prompt(self, idx, prompt):
        if self.dataset == 'gsm8k':
            if idx not in self.attacker_idxes: 
                prompt += PI_USER_PROMPT_RE_GSM8K
            else: 
                prompt += PI_ATTACKER_USER_PROMPT_RE_GSM8K
        else:
            if idx not in self.attacker_idxes: 
                prompt += PI_USER_PROMPT_RE
            else: 
                prompt += PI_ATTACKER_USER_PROMPT_RE
        return prompt
    
    def _get_ta_regenerate_prompt(self, idx, prompt):
        if idx not in self.attacker_idxes: 
            prompt += TA_USER_PROMPT_RE
        else: 
            prompt = TA_ATTACKER_USER_PROMPT_RE
        return prompt

    def re_generate_agent(self, idx, prompt):
        if self.attack_mode == "MA":
            prompt = self._get_ma_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "PI":
            prompt = self._get_pi_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "TA":
            prompt = self._get_ta_regenerate_prompt(idx, prompt)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
        
        response = self.agents[idx].chat(prompt)
        return idx, response

    def re_generate(self):
        prompts = []
        for idx in range(self.num_agents):
            views = {}
            prompt = ""
            in_edges = self.adj_matrix[:, idx]
            in_idxs = np.nonzero(in_edges)[0]
            
            if len(in_idxs) > 0:
                for in_idx in in_idxs:
                    agent = self.agents[in_idx]
                    if self.attack_mode == "TA":
                        prompt += f"Agent_{in_idx}'s reply: {agent.last_response} \n"  
                    else:
                        views[f"Agent_{in_idx}'s View:"] = {f"Agent_{in_idx}'s answer": agent.last_response['answer'],
                                                            f"Agent_{in_idx}'s reason": agent.last_response['reason']}
                
                if self.attack_mode != "TA":
                    prompt += str(views)
            else:
                prompt += "No responses from other agents.\n"

            prompts.append(prompt)
        
        agent_responses = []
        for idx in range(self.num_agents):
            response = self.re_generate_agent(idx, prompts[idx])
            agent_responses.append(response)
        return agent_responses
    
    async def are_generate_agent(self, idx, prompt):
        if self.attack_mode == "MA":
            prompt = self._get_ma_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "PI":
            prompt = self._get_pi_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "TA":
            prompt = self._get_ta_regenerate_prompt(idx, prompt)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
        
        response = await self.agents[idx].achat(prompt)
        return idx, response

    async def are_generate(self):
        prompts = []
        for idx in range(self.num_agents):
            views = {}
            prompt = "observation" if self.attack_mode == "TA" else ""
            in_edges = self.adj_matrix[:, idx]
            in_idxs = np.nonzero(in_edges)[0]
            
            if len(in_idxs) > 0:
                for in_idx in in_idxs:
                    agent = self.agents[in_idx]
                    if self.attack_mode == "TA":
                        prompt += f"Agent_{in_idx}'s reply: {agent.last_response} \n"  
                    else:
                        views[f"Agent_{in_idx}'s View:"] = {f"Agent_{in_idx}'s answer": agent.last_response['answer'],
                                                            f"Agent_{in_idx}'s reason": agent.last_response['reason']}
                
                if self.attack_mode != "TA":
                    prompt += str(views)
            else:
                prompt += "No responses from other agents.\n"

            prompts.append(prompt)
        
        tasks = []
        for idx in range(self.num_agents):
            tasks.append(asyncio.create_task(self.are_generate_agent(idx, prompts[idx])))
        agent_responses = await asyncio.gather(*tasks)
        return agent_responses


class AgentGraphWithDefense: 
    def __init__(self, adj_matrix, system_prompts, attacker_idxes, model_type="gpt-4o-mini", attack_mode: str = "MA", dataset='gsm8k'): 
        self.num_agents = len(system_prompts)
        self.adj_matrix = adj_matrix
        self.system_prompts = system_prompts
        self.attacker_idxes = attacker_idxes
        self.model_type = model_type
        self.attack_mode = attack_mode
        self.agents = []
        self.dataset = dataset
        for i in range(len(adj_matrix)): 
            self.agents.append(Agent(system_prompts[i], model_type, attack_mode))


    def _get_ma_first_prompt(self, idx, query, context, ):
        if idx not in self.attacker_idxes: 
            prompt = MA_USER_PROMPT_FIRST.format(query=query)
        else: 
            prompt = MA_ATTACKER_SYS_PROMPT_FIRST.format(query=query, context=context)
        return prompt
    
    def _get_pi_first_prompt(self, idx, qa_data, ):
        if self.dataset == 'gsm8k':
            if idx not in self.attacker_idxes: 
                prompt = PI_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0])
            else:                 
                prompt = PI_ATTACKER_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0], qa_data_2=qa_data[2])
        else:
            if idx not in self.attacker_idxes: 
                prompt = PI_USER_PROMPT_FIRST.format(qa_data=qa_data[0])
            else: 
                prompt = PI_ATTACKER_USER_PROMPT_FIRST.format(qa_data=qa_data[0], qa_data_2=qa_data[2])
        return prompt
    
    def _get_ta_first_prompt(self, idx, user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad, ):
        if idx not in self.attacker_idxes: 
            prompt = TA_USER_PROMPT.format(available_tool_descriptions = available_tool_descriptions, input = user_instruction, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
        else: 
            prompt = TA_ATTACKER_USER_PROMPT.format(input_text=agent_scratchpad)
        return prompt
    
    
    def first_generate_agent(self, idx, *args, ):
        if self.attack_mode == "MA":
            query, context = args
            if idx not in self.attacker_idxes: 
                prompt = MA_USER_PROMPT_FIRST.format(query=query)
            else: 
                prompt = MA_ATTACKER_SYS_PROMPT_FIRST.format(query=query, context=context)
        
        elif self.attack_mode == "PI":
            qa_data = args[0]
            if self.dataset == 'gsm8k':
                if idx not in self.attacker_idxes: 
                    prompt = PI_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0])
                else:                 
                    prompt = PI_ATTACKER_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0], qa_data_2=qa_data[2])
            else:
                if idx not in self.attacker_idxes: 
                    prompt = PI_USER_PROMPT_FIRST.format(qa_data=qa_data[0])
                else: 
                    prompt = PI_ATTACKER_USER_PROMPT_FIRST.format(qa_data=qa_data[0], qa_data_2=qa_data[2])
        
        elif self.attack_mode == "TA":
            user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad = args[0]
            if idx not in self.attacker_idxes: 
                prompt = TA_USER_PROMPT.format(available_tool_descriptions = available_tool_descriptions, input = user_instruction, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
            else: 
                prompt = TA_ATTACKER_USER_PROMPT.format(input_text=agent_scratchpad)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
        
        response = self.agents[idx].chat(prompt)
        return idx, response
    
    def first_generate(self, *args, ):
        agent_responses = []
        for idx in range(self.num_agents): 
            response = self.first_generate_agent(idx, *args, )
            agent_responses.append(response)
        
        return agent_responses
    
    async def afirst_generate_agent(self, idx, *args, ):
        if self.attack_mode == "MA":
            query, context = args
            if idx not in self.attacker_idxes: 
                prompt = MA_USER_PROMPT_FIRST.format(query=query)
            else: 
                prompt = MA_ATTACKER_SYS_PROMPT_FIRST.format(query=query, context=context)
        
        elif self.attack_mode == "PI":
            qa_data = args[0]
            if self.dataset == 'gsm8k':
                if idx not in self.attacker_idxes: 
                    prompt = PI_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0])
                else:                     
                    prompt = PI_ATTACKER_USER_PROMPT_FIRST_GSM8K.format(qa_data=qa_data[0], qa_data_2=qa_data[2])
            else:
                if idx not in self.attacker_idxes: 
                    prompt = PI_USER_PROMPT_FIRST.format(qa_data=qa_data[0])
                else: 
                    prompt = PI_ATTACKER_USER_PROMPT_FIRST.format(qa_data=qa_data[0], qa_data_2=qa_data[2])

        elif self.attack_mode == "TA":
            user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad = args[0]
            if idx not in self.attacker_idxes: 
                prompt = TA_USER_PROMPT.format(available_tool_descriptions = available_tool_descriptions, input = user_instruction, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
            else: 
                prompt = TA_ATTACKER_USER_PROMPT.format(input_text=agent_scratchpad)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
            
        response = await self.agents[idx].achat(prompt)
        return idx, response
    
    async def afirst_generate(self, *args, ): 
        tasks = []
        for idx in range(self.num_agents): 
            tasks.append(asyncio.create_task(self.afirst_generate_agent(idx, *args, )))
        agent_responses = await asyncio.gather(*tasks)
        return agent_responses
        

    def _get_ma_regenerate_prompt(self, idx, prompt):
        if idx not in self.attacker_idxes: 
            prompt += MA_USER_PROMPT_RE
        else: 
            prompt += MA_ATTACKER_USER_PROMPT_RE
        return prompt
    
    def _get_pi_regenerate_prompt(self, idx, prompt):
        if self.dataset == 'gsm8k':
            if idx not in self.attacker_idxes: 
                prompt += PI_USER_PROMPT_RE_GSM8K
            else: 
                prompt += PI_ATTACKER_USER_PROMPT_RE_GSM8K
        else:
            if idx not in self.attacker_idxes: 
                prompt += PI_USER_PROMPT_RE
            else: 
                prompt += PI_ATTACKER_USER_PROMPT_RE
        return prompt
    
    def _get_ta_regenerate_prompt(self, idx, prompt):
        if idx not in self.attacker_idxes: 
            prompt += TA_USER_PROMPT_RE
        else: 
            prompt = TA_ATTACKER_USER_PROMPT_RE
        return prompt

    def re_generate_agent(self, idx, prompt):
        if self.attack_mode == "MA":
            prompt = self._get_ma_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "PI":
            prompt = self._get_pi_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "TA":
            prompt = self._get_ta_regenerate_prompt(idx, prompt)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
        
        response = self.agents[idx].chat(prompt)
        return idx, response

    def re_generate(self):
        prompts = []
        for idx in range(self.num_agents):
            views = {}
            prompt = ""
            in_edges = self.adj_matrix[:, idx]
            in_idxs = np.nonzero(in_edges)[0]
            
            if len(in_idxs) > 0:
                for in_idx in in_idxs:
                    agent = self.agents[in_idx]
                    if agent.get_role() == "normal": 
                        if self.attack_mode == "TA":
                            prompt += f"Agent_{in_idx}'s reply: {agent.last_response} \n"  
                        else:
                            views[f"Agent_{in_idx}'s View:"] = {f"Agent_{in_idx}'s answer": agent.last_response['answer'],
                                                                f"Agent_{in_idx}'s reason": agent.last_response['reason']}
                
                if self.attack_mode != "TA":
                    prompt += str(views)
            else:
                prompt += "No responses from other agents.\n"

            prompts.append(prompt)
        
        agent_responses = []
        for idx in range(self.num_agents):
            response = self.re_generate_agent(idx, prompts[idx])
            agent_responses.append(response)
        return agent_responses

    async def are_generate_agent(self, idx, prompt):
        if self.attack_mode == "MA":
            prompt = self._get_ma_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "PI":
            prompt = self._get_pi_regenerate_prompt(idx, prompt)
        elif self.attack_mode == "TA":
            prompt = self._get_ta_regenerate_prompt(idx, prompt)
        else:
            raise ValueError(f"Unsupported attack mode: {self.attack_mode}")
        
        response = await self.agents[idx].achat(prompt)
        return idx, response

    async def are_generate(self):
        prompts = []
        for idx in range(self.num_agents):
            views = {}
            prompt = "observation" if self.attack_mode == "TA" else ""
            in_edges = self.adj_matrix[:, idx]
            in_idxs = np.nonzero(in_edges)[0]
            
            if len(in_idxs) > 0:
                for in_idx in in_idxs:
                    agent = self.agents[in_idx]
                    if agent.get_role() == "normal": 
                        if self.attack_mode == "TA":
                            prompt += f"Agent_{in_idx}'s reply: {agent.last_response} \n"  
                        else:
                            views[f"Agent_{in_idx}'s View:"] = {f"Agent_{in_idx}'s answer": agent.last_response['answer'],
                                                                f"Agent_{in_idx}'s reason": agent.last_response['reason']}
                
                if self.attack_mode != "TA":
                    prompt += str(views)
            else:
                prompt += "No responses from other agents.\n"

            prompts.append(prompt)
        
        tasks = []
        for idx in range(self.num_agents):
            tasks.append(asyncio.create_task(self.are_generate_agent(idx, prompts[idx])))
        agent_responses = await asyncio.gather(*tasks)
        return agent_responses
