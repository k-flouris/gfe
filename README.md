
# Gradient-flow encoding

The autoencoder model typically uses an encoder to map data to a lower dimensional latent space and a decoder to reconstruct it. However, relying on an encoder for inversion can lead to suboptimal representations, particularly limiting in physical sciences where precision is key. We introduce a decoder-only method using gradient flow to directly encode data into the latent space, defined by ordinary differential equations (ODEs). This approach eliminates the need for approximate encoder inversion. We train the decoder via the adjoint method and show that costly integrals can be avoided with minimal accuracy loss. Additionally, we propose a 2nd order ODE variant, approximating Nesterov's accelerated gradient descent for faster convergence. To handle stiff ODEs, we use an adaptive solver that prioritizes loss minimization, improving robustness. Compared to traditional autoencoders, our method demonstrates explicit encoding and superior data efficiency, which is crucial for data-scarce scenarios in the physical sciences. Furthermore, this work paves the way for integrating machine learning into scientific workflows, where precise and efficient encoding is critical.

https://arxiv.org/abs/2412.00864

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Main Features](#main-features)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

---

## Installation

1. Clone this repository:
   ```bash
   gh repo clone k-flouris/gfe
   ```
   or
    ```bash
     git clone https://github.com/k-flouris/gfe.git
    ```
   
2. Install required dependencies (see [Dependencies](#dependencies)).

---

## Project Structure

```
.
├── gflow.py               # Implements gradient flow dynamics
├── macros.py              # Contains utility macros and reusable functions
├── main.py                # Entry point for running the project
├── networks_gen_grad.py   # Defines neural network models with gradient-based generation
├── networks_gen.py        # Implements additional generative network architectures
├── networks.py            # Implements neural network architectures
├── plotting.py            # Handles data visualization and plotting
├── tsnedeom.py            # TSNE-based dimensionality reduction algorithms
├── visualisation.py       # Additional visualization utilities
```

---

## Usage

Run the project by executing `main.py`:
```bash
python main.py -c <config_file>
```

### Parameters

- `-c <config_file>`: Specify a configuration file to customize the experiment.
- Additional options and hyperparameters can be specified in the configuration files.

---

## Configuration

The project relies on configuration files to define experimental setups. These configurations typically include:

- Model hyperparameters (e.g., learning rate, architecture details)
- Data preprocessing settings
- Training options (e.g., epochs, batch size)
- Visualization options (e.g., plots to generate)

To customize configurations, edit the relevant `.json` or `.yaml` file (if applicable) and pass it to the `main.py` script using the `--config` argument.

---

## Main Features

1. **Gradient Flow Dynamics**:
   - Implements gradient flow-based algorithms for neural networks and other systems.

2. **Customizable Neural Networks**:
   - Modular design for creating and training various network architectures.

3. **Visualization**:
   - Comprehensive plotting capabilities using `plotting.py` and `visualisation.py`.
   - Supports TSNE and other dimensionality reduction methods via `tsnedeom.py`.

4. **Extensible Framework**:
   - Add new models and visualizations with minimal changes to existing code.

---

## Dependencies

The project requires the following Python libraries:

- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`
- `torch` (for neural network models)
- `seaborn`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Contributing

Contributions are welcome! If you have ideas for improving the project, feel free to submit a pull request or open an issue.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

---
