import os
from typing import Generator

import openai

import nebuly

nebuly.init(api_key="ciao")

for chunk in openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Say this is a test",
    max_tokens=7,
    temperature=0,
    stream=True,
):
    print(chunk)
