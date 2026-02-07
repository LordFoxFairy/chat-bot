from langchain_core.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Useful for math calculations. Input should be a mathematical expression."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str, **kwargs: Any) -> str:
        """Evaluate the math expression."""
        try:
            # WARNING: eval is unsafe in production environments!
            # Using it here only for demonstration purposes.
            # In a real app, use a safer math parser like `numexpr` or limited eval.
            return str(eval(expression))
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"

    async def _arun(self, expression: str, **kwargs: Any) -> str:
        """Asynchronously evaluate the math expression."""
        return self._run(expression)
