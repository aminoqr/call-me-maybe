"""Data models for the function calling system."""

from pydantic import BaseModel
from typing import Dict, Any


class Parameter(BaseModel):
    """Schema for a function parameter type."""

    type: str


class FunctionDefinition(BaseModel):
    """Definition of a callable function."""

    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Dict[str, str]


class PromptInput(BaseModel):
    """Input prompt for function calling."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Validated result of a function call."""

    prompt: str
    name: str
    parameters: Dict[str, Any]
