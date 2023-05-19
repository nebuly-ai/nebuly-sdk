# Nebuly SDK
The SDK for instrumenting applications for tracking AI costs.

## Supported Providers
    - OpenAI
    - Azure OpenAI

## Usage

```python
import os
import nebuly
... 

if __name__ == "__main__":
    nebuly.api_key = os.getenv("NEBULY_API_KEY")
    nebuly.init("my_project_name", nebuly.DevelopmentPhase.EXPERIMENTATION)
    ... 
```

### Tagging Specific Methods
```python
import os
import nebuly
... 

@nebuly.tracker(
    project="my_project_2",
    phase=nebuly.DevelopmentPhase.PRODUCTION,
    Task=nebuly.Task.TEXT_SUMMARIZATION,
)
def my_method(self, *args, **kwargs):
    # My method to be tagged differently
    ...

...

if __name__ == "__main__":
    nebuly.api_key = os.getenv("NEBULY_API_KEY")
    nebuly.init("my_project_name", nebuly.DevelopmentPhase.EXPERIMENTATION)
    ...
```
### Tagging Specific Code-Sections
```python
import os
import nebuly
... 
    
def my_method(self, *args, **kwargs):
    ...
    # Some Code
    ...
    with nebuly.tracker(
        project="my_project_2",
        phase=nebuly.DevelopmentPhase.PRODUCTION,
        Task=nebuly.Task.TEXT_SUMMARIZATION,
    ):
        # Specific code section to be tagged differently
        ...
    ...
    # Some other code
    ... 

...

if __name__ == "__main__":
    nebuly.api_key = os.getenv("NEBULY_API_KEY")
    nebuly.init("my_project_name", nebuly.DevelopmentPhase.EXPERIMENTATION)
    ...
```