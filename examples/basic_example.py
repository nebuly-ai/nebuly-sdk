import asyncio
import os

from nebuly.init import init

API_KEY = str(os.environ.get("NEBULY_API_KEY"))

init(api_key=API_KEY)

import openai  # noqa  # pylint: disable=wrong-import-position, wrong-import-order

OPENAI_KEY = str(os.environ.get("OPENAI_KEY"))
openai.api_key = OPENAI_KEY


def open_ai_completion() -> None:
    response = openai.Completion.create(  # type: ignore
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
    )

    print(response)

    response = openai.Completion.create(  # type: ignore
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
        stream=True,
    )

    print(list(response))


async def open_ai_completion_async() -> None:
    response = await openai.Completion.acreate(  # type: ignore
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
    )

    print(response)

    response = await openai.Completion.acreate(  # type: ignore
        model="text-davinci-003",
        prompt="Say this is a test",
        max_tokens=7,
        temperature=0,
        stream=True,
    )

    async for item in response:
        print(item)


def open_ai_chat() -> None:
    completion = openai.ChatCompletion.create(  # type: ignore
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0,
    )

    print(completion)

    completion = openai.ChatCompletion.create(  # type: ignore
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0,
        stream=True,
    )

    print(list(completion))


async def open_ai_chat_async() -> None:
    completion = await openai.ChatCompletion.acreate(  # type: ignore
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0,
    )

    print(completion)

    completion = await openai.ChatCompletion.acreate(  # type: ignore
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        temperature=0,
        stream=True,
    )

    async for item in completion:
        print(item)


def open_ai_embedding() -> None:
    result = openai.Embedding.create(  # type: ignore
        model="text-embedding-ada-002", input="The food was delicious and the waiter..."
    )

    print(result)


async def open_ai_embedding_async() -> None:
    result = await openai.Embedding.acreate(  # type: ignore
        model="text-embedding-ada-002", input="The food was delicious and the waiter..."
    )

    print(result)


async def main() -> None:
    open_ai_completion()
    await open_ai_completion_async()
    open_ai_chat()
    await open_ai_chat_async()
    open_ai_embedding()
    await open_ai_embedding_async()


if __name__ == "__main__":
    asyncio.run(main())
