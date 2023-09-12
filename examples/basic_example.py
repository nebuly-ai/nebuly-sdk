import asyncio
import os

from nebuly.entities import DevelopmentPhase
from nebuly.init import init

API_KEY = str(os.environ.get("NEBULY_API_KEY"))

init(api_key=API_KEY, project="my_project", phase=DevelopmentPhase.EXPERIMENTATION)

import openai

OPENAI_KEY = str(os.environ.get("OPENAI_KEY"))
openai.api_key = OPENAI_KEY


def open_ai_completion():
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
    )

    print(response)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
        stream=True,
    )

    print(list(response))


async def open_ai_completion_async():
    response = await openai.Completion.acreate(
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
    )

    print(response)

    response = await openai.Completion.acreate(
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
        stream=True,
    )

    async for item in response:
        print(item)


def open_ai_chat():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(completion)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        stream=True,
    )

    print(list(completion))


async def open_ai_chat_async():
    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    print(completion)

    completion = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        stream=True,
    )

    async for item in completion:
        print(item)


async def main():
    open_ai_completion()
    await open_ai_completion_async()
    open_ai_chat()
    await open_ai_chat_async()


if __name__ == "__main__":
    asyncio.run(main())
