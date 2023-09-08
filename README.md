# Nebuly

## TODO

1. Nice to have: semantic versioning expansion in package (maybe OSS library or Poetry?)
1. Check the publisher doens't crash, and if it does re start it somehow
1. batch processing

Make sure we publish all these fields

- project name
- user
- provider
- library
- version
- args, kwars
- returned values

## Design

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
