from pydantic import BaseModel
from typing import Dict, Any


class Parameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]


class PromptInput(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
