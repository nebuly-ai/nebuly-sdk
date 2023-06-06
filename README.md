# Nebuly SDK
The SDK for instrumenting applications for tracking AI costs.

## Code Quality Checks
This section provides guidelines and instructions on how to perform code quality checks using [**Black**](https://github.com/psf/black) as a formatter, [**Ruff**](https://github.com/charliermarsh/ruff) as a linter, and [**SonarCloud**](https://www.sonarsource.com/products/sonarcloud/) as a code quality checker. These tools assist in ensuring that the codebase adheres to a consistent style, follows best practices, and meets predefined quality standards.

### **Setup**

To set up the code quality checks for this project:

1. Clone the repository
2. Install **`pre-commit`**
```
brew install pre-commit
```
3. Run the setup command to install the necessary requirements, including Poetry for handling dependencies
```
make setup
```
### **Code Formatting and Linting**

The code formatting and linting checks help maintain consistent style and identify potential issues. Black and Ruff are automatically invoked with each commit, but they can also be utilized independently without committing changes:

- To run Black alone
```
black .
```
- To display the issues detected by the linter
```
make lint
```
- To automatically apply the formatter changes and the suggested changes by the linter, use the following command
```
make lint-fix
```
### **SonarCloud**

SonarCloud performs advanced code analysis to detect bugs, vulnerabilities, and code smells. It is triggered by pull requests (PRs). To view the report generated by SonarCloud:

1. [Log in](https://sonarcloud.io/login) to SonarCloud using your GitHub account
2. Navigate to the PR where you want to view the report

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