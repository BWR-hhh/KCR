# Implementation of "Permitted Knowledge Boundary: Evaluating the Knowledge-Constrained Responsiveness of Large Language Models"

This project implements the algorithms and methodologies described in the research paper "[Permitted Knowledge Boundary: Evaluating the Knowledge-Constrained Responsiveness of Large Language Models]". The paper introduces an innovative benchmark to evaluate the Knowledge-Constrained Responsiveness of Large Language Models. This repository provides a practical, open-source implementation to replicate the results and facilitate further research or application.

## Table of Contents

- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)


## Usage

To replicate the experiments from the paper:

1. **Prepare the Data**:
   - Download the dataset from [link] and place it in `data/` directory.

2. **Run the Main Script**:
   ```bash
   python flashrag-pipeline-3fold-new.py
   ```

## Configuration

Adjust the experiment settings in `config.yaml`:

```python
--generator gemma-2-9b # mistral-7B llama3-8B vicuna-7b glm-4-9b qwen-7B-chat gemma-2-9b qwen3-8b
--prompt_mode naive # "naive", "kbllm", "rag"
--dataset simpleqa # simpleqa webnlg

```

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). See the [LICENSE](LICENSE) file for details.
