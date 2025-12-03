## Installed via
- uv init
- uv add torch
- uv add matplotlib

## Launch
- uv run re8_XOR_SOLVER.py

---

# XOR Neural Network from Scratch in PyTorch

![Alt-text](/Capture.PNG)

![Alt-text](/Capture2.PNG)

This project implements a **simple feedforward neural network** from scratch using **PyTorch tensors** (without relying on `torch.nn` modules for layers). 

- Manual implementation of **linear layers**.
- **Leaky ReLU** activation function.
- **Mean Squared Error (MSE)** loss.
- Custom **AdamW optimizer** with weight decay and momentum.
- Training the network to solve the classic **XOR problem**.
- Plotting training loss using **Matplotlib**.


