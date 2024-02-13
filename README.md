# Nebuly SDK

The SDK for instrumenting applications for tracking AI costs.


### **Setup**

To set up the code quality checks for this project:

1. Clone the repository
1. Run the setup command to install the necessary requirements, including Poetry for
   handling dependencies

```
make setup
```

### **Code Formatting and Linting**

The code formatting and linting checks help maintain consistent style and identify
potential issues. Black and Ruff are automatically invoked with each commit, but they
can also be utilized independently without committing changes:

- To display the issues detected by the linter

```
make lint
```

- To automatically apply the formatter changes and the suggested changes by the linter,
  use the following command

```
make lint-fix
```

## Supported Providers

    - OpenAI
    - Azure OpenAI
    - Cohere
    - Anthropic
    - HuggingFace pipelines
    - HuggingFace HUB
    - LangChain
    - LlamaIndex
    - Amazon Bedrock
    - Amazon SageMaker
    - Google PALM API
    - Google VertexAI

## Usage

Make sure you initialize Nebuly before importing other libraries
like `openai`, `cohere`, `huggingface`, etc.

### Simple usage

In the simple case, you can just import nebuly and call the init function with your API
key. This will automatically
setup all the tracking for you. After that, you can call the other libraries as normal.

#### Example with OpenAI

```python
import os
import nebuly

api_key = os.getenv("NEBULY_API_KEY")
nebuly.init(api_key=api_key)

import os
from openai import OpenAI

client = OpenAI()
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
    user_id="user-123",
    feature_flags=["new-feature_flag"],
)
```

### Advanced usage: Context managers

In the simple case, each call will be stored as a separate Interaction, you can use
context managers to group
more calls in a single Interaction:

#### Example with OpenAI and Cohere

```python
import os
import nebuly
from nebuly.contextmanager import new_interaction

api_key = os.getenv("NEBULY_API_KEY")
nebuly.init(api_key=api_key)

# Setup OpenAI
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup Cohere
import cohere

co = cohere.Client(os.getenv("COHERE_API_KEY"))

with new_interaction(user_id="test_user", user_group_profile="test_group") as interaction:
    # interaction.set_input("Some custom input")
    # interaction.set_history([{"role": "user/assistant", "content": "sample content"}}])
    completion_1 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an helpful assistant"},
            {"role": "user", "content": "Hello world"}
        ]
    )
    completion_2 = co.generate(
        prompt='Please explain to me how LLMs work',
    )
    # interaction.set_output("Some custom output")
```

## LangChain Callbacks

```python
import os

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from nebuly.providers.langchain import LangChainTrackingHandler

callback = LangChainTrackingHandler(
    user_id="test_user",
    api_key=os.getenv("NEBULY_API_KEY"),
)

llm = ChatOpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(
    "colorful socks",
    callbacks=[callback],
)
```

## LlamaIndex Callbacks

```python
import os
from nebuly.providers.llama_index import LlamaIndexTrackingHandler

handler = LlamaIndexTrackingHandler(
    api_key=os.getenv("NEBULY_API_KEY"), user_id="test_user"
)

import llama_index
from llama_index import SimpleDirectoryReader, VectorStoreIndex

llama_index.global_handler = handler

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
```

## Variants Usage

```python
from nebuly.ab_testing import ABTesting

client = ABTesting("your_nebuly_api_key")

variants = client.get_variants(
  user="<user_id>",
  feature_flags=["feature_flag_a", "feature_flag_b"]
)
print(variants)
```
