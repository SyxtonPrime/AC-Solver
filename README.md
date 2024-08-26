Here's a draft for the README file:

---

# AC-Env-Solver

## What Makes Math Problems Hard for Reinforcement Learning Agents: A Case Study

### Overview

This repository accompanies the paper *"What Makes Math Problems Hard for Reinforcement Learning Agents: A Case Study."* It includes an implementation of the AC Environment in Gymnasium, two classical search algorithms (BFS and Greedy Search), and a PPO agent that works within this environment. Additionally, the repository contains Jupyter notebooks for reproducing the analyses and figures presented in the paper.

### Installation

To work with the AC Environment or build upon it, you can simply install the package using pip:

```bash
pip install ac_env_solver
```

If you wish to reproduce the plots and analyses in the paper, you will need to clone the repository locally. Here is the recommended process:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/AC-Env-Solver.git
   cd AC-Env-Solver
   ```

2. Install the package locally using Poetry:

   ```bash
   poetry install
   ```

   Alternatively, you can install the package using pip:

   ```bash
   pip install .
   ```

### Usage

After installation, you can start using the environment and agents as follows:

#### Initializing the AC Environment

```python
from ac_env_solver.envs.ac_env import ACEnv

acenv = ACEnv()
```

#### Solving the Environment with PPO

```python
from ac_env_solver.agents.ppo import train_ppo

train_ppo()
```

#### Performing Classical Search

Specify a presentation and perform a greedy search:

```python
presentation = [1, 1, -2, -2, -2, 0, 0, 1, 2, 1, -2, -1, -2, 0]

from ac_env_solver.search.greedy import greedy_search

greedy_search(presentation)
```

### Notebooks

The `notebooks/` directory contains Jupyter notebooks that reproduce the figures and results discussed in the paper:

- **`Sections-1-3-and-4.ipynb`**: Reproduces figures in Sections 1, 3, and 4 of the paper.
- **`Section-5.ipynb`**: Reproduces figures in Section 5 of the paper.
- **`Stable-AK3.ipynb`**: Provides code demonstrating that AK(3) is a stably AC-trivial presentation, a major result of the paper.

To run these notebooks, you must clone the repository locally as described above.

### Contributing

Contributions to this project are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Run tests with `pytest` and ensure your code is formatted with `black`:

   ```bash
   poetry run pytest
   poetry run black .
   ```

4. Submit a pull request with a clear description of your changes.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to adjust or expand upon this as needed!