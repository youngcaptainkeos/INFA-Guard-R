from openai.types.chat import ChatCompletion
from ray.serve.handle import DeploymentHandle


class LocalChatCompletions:
    def __init__(self, handle: DeploymentHandle) -> None:
        self.handle = handle

    async def create(self, **kwargs) -> ChatCompletion:
        result_dict = await self.handle.generate.remote(**kwargs)
        return ChatCompletion.model_validate(result_dict)


class LocalChat:
    def __init__(self, handle: DeploymentHandle) -> None:
        self.completions = LocalChatCompletions(handle)


class AsyncLocalModelClient:
    def __init__(self, handle: DeploymentHandle) -> None:
        self.chat = LocalChat(handle)

