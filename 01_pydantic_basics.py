"""Pydantic warm-up. Run: python 01_pydantic_basics.py"""
from typing import List
from pydantic import BaseModel, Field, ValidationError, field_validator


class Paper(BaseModel):
    title: str
    url: str
    year: int = Field(ge=1990, le=2030)
    authors: List[str] = []

    @field_validator("url")
    @classmethod
    def must_be_http(cls, v: str) -> str:
        if not v.startswith("http"):
            raise ValueError("url must start with http")
        return v


p = Paper(
    title="Attention Is All You Need",
    url="https://arxiv.org/abs/1706.03762",
    year=2017,
)
print("valid paper  ->", p.title, p.year)
print("model_dump() ->", p.model_dump())

try:
    Paper(title="Oops", url="ftp://nope", year="not a year")
except ValidationError as e:
    print("\nValidationError caught:\n", e)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    extractor = llm.with_structured_output(Paper)     # LLM -> validated Paper
    result = extractor.invoke(
        "Extract the paper: 'Attention Is All You Need', "
        "Vaswani et al., 2017, https://arxiv.org/abs/1706.03762"
    )
    print("\nLLM returned a validated Paper object:\n", result)