# Nebuly

## TODO

1. Thread for consuming events instead of calling requests from the thread that generated the event
2. Handle args and kwargs serialization - what happen if the args are not serializable?
3. Context manager for patching single portion of codes
   - nested context managers should have "stack" behaviour
4. Nice to have: semantic versioning expansion in package (maybe OSS library or Poetry?)

## Current state

```mermaid
classDiagram
    WrappingStrategy <|-- APICallWrappingStrategy
    WrappingStrategy <|-- GeneratorWrappingStrategy
    GeneratorWrappingStrategy <|-- TextCompletionStrategy
    GeneratorWrappingStrategy <|-- ChatCompletionStrategy
    DataPackageConverte <|-- OpenAIDataPackageConverter
    Track --> WrappingStrategy
    DataPackageConverte --> WrappingStrategy
    Track <|-- OpenAITracker
    class OpenAITracker{
        completion_create
        chat_completion_create
        edit_create
        image_create
        image_edit
        image_variation
        embedding_create
        audio_transcribe
        audio_translate
        moderation_create
        finetune
    }
```

### Issues

- Logic and configs are mixed
- Potentially lose information
    - When new api versions
    - When api changes
    - When a bug is present


## New state

```mermaid
classDiagram
    Package --> Patcher
    Observer ..* Patcher
    Patcher --> Watched
    Observer --> Watched
    Patcher --> OpenAI
    class Patcher{
        observer
    }
    class Package{
        name: str
        versions: list[str]
        to_patch: list[str]
    }
    class Watched{
        function: Callable
        called_at: datetime
        called_with_args: tuple
        called_with_kwargs: dict[str, Any]
        returned: Any
    }
    class Observer{
        proyect: str
        user: str
        phase: str
        watch()
    }
```

### Advantages

- Every monkey patching is the same
- Easier to add more libraries and support different versions
- In case of a bug we don't lose any data, we can go back in time and reprocess


## TODO

Make sure we publish all these fields

- project name
- user
- provider
- library
- version
- args, kwars
- returned values
