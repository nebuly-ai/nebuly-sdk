# Nebuly SDK

The SDK for instrumenting applications for tracking AI costs.

### **Setup**

To set up the code quality checks for this project:

1. Clone the repository
2. Install **`pre-commit`**

```
brew install pre-commit
```

3. Run the setup command to install the necessary requirements, including Poetry for
   handling dependencies

```
make setup
```

### **Code Formatting and Linting**

The code formatting and linting checks help maintain consistent style and identify
potential issues. Black and Ruff are automatically invoked with each commit, but they
can also be utilized independently without committing changes:

- To run Black alone

```
black .
```

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
    development_phase=nebuly.DevelopmentPhase.PRODUCTION,
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
        development_phase=nebuly.DevelopmentPhase.PRODUCTION,
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