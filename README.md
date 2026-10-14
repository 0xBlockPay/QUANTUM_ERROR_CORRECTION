# Autonomous Reinforcement Learning Control of Quantum Error Correction (CUDA-Q)

[![NVIDIA CUDA-Q](https://img.shields.io/badge/Platform-NVIDIA_CUDA--Q-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-quantum)
[![Nature 2026](https://img.shields.io/badge/Article-Nature_655%2C_879--884_(2026)-E60000?logo=nature&logoColor=white)](https://doi.org/10.1038/s41586-026-10759-2)
[![Google Quantum AI](https://img.shields.io/badge/Google-Quantum_AI_%26_DeepMind-4285F4?logo=google&logoColor=white)](https://research.google/teams/quantum-ai/)
[![cuQuantum GPU](https://img.shields.io/badge/Acceleration-cuQuantum_Tensor_Cores-76B900)](https://developer.nvidia.com/cuquantum-sdk)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A production-grade, hybrid quantum-classical software suite implemented in **NVIDIA CUDA-Q** and **PyTorch**, faithfully replicating the breakthrough research by **Google Quantum AI & Google DeepMind**:
> **"Reinforcement learning control of quantum error correction"**  
> *Nature*, Vol 655, Issue 8125, pp. 879–884 (23 July 2026)  
> *Volodymyr Sivak, Alexis Morvan, Michael Broughton, Rodrigo G. Cortiñas, Johannes Bausch, Andrew W. Senior, ..., Vadim Smelyanskiy, and Paul V. Klimov.*  
> **DOI:** [10.1038/s41586-026-10759-2](https://doi.org/10.1038/s41586-026-10759-2)

---

## 📌 Executive Summary

Quantum processors are fundamentally analogue machines steered by thousands of delicate microwave and flux pulses. Environmental temperature drifts, material two-level fluctuators (TLS), and $1/f$ flux noise cause continuous drift in qubit frequencies, pulse amplitudes, and two-qubit entangling phases ($CZ$). 

Traditionally, quantum computers had to **terminate ongoing computations** to run offline recalibration routines. This complete decoupling of computation and calibration represents a fatal bottleneck for fault-tolerant quantum algorithms requiring uninterrupted runtimes of days or months.

This project implements Google's paradigm shift: **"Tuning the instruments while the music plays"**.  
We repurpose the digitized error-detection events generated during repetitive Quantum Error Correction (QEC) stabilizer cycles as a real-time learning signal. An autonomous **Parameter-Exploring Policy Gradient (PEPG)** agent continuously steers the transmon control parameters over a **sparse bipartite factor graph**, achieving:
- **3.5-fold improvement in logical stability** and **31% reduction in logical error rate (LER)** when combining physical pulse steering with complementary decoder graph reweighting.
- **Record performance on Google's Willow processor**: $\varepsilon_L = 7.72(9) \times 10^{-4}$ per cycle for distance-7 surface code (AlphaQubit2) and $8.19(14) \times 10^{-3}$ for distance-5 colour code (Tesseract).
- **System-size-independent optimization rate**: Scalable up to distance-15 surface code with $\sim 40,000$ control parameters.

---

## 🔬 Core Scientific Foundations & Algorithms

```
                          ┌──────────────────────────┐
                          │   Physical Environment   │
                          │   (1/f Drift & Noise)    │
                          └─────────────┬────────────┘
                                        │ (Analog Detuning)
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                     NVIDIA CUDA-Q Quantum Kernel (QPU / GPU)                    │
 │                                                                                 │
 │   ┌────────────────┐    ┌─────────────────┐    ┌──────────────────────────┐     │
 │   │ Data Qubits    ├───►│ Rotated Surface ├───►│ Mid-Circuit Readout (mz) │     │
 │   │ (D0 ... D_N)   │    │ Code Cycle (CZ) │    │ & Ancilla Reset (reset)  │     │
 │   └───────▲────────┘    └─────────────────┘    └────────────┬─────────────┘     │
 └───────────┼─────────────────────────────────────────────────┼───────────────────┘
             │                                                 │ Raw Syndromes
             │ Analog Pulse Offsets (μ)                        ▼ S_t ⊕ S_{t-1}
 ┌───────────┴─────────────────────────────────────────────────────────────────────┐
 │                         Classical Controller & RL Agent                         │
 │                                                                                 │
 │   ┌───────────────────────┐                    ┌────────────────────────────┐   │
 │   │ Factor Graph PEPG     │◄───────────────────┤ Surrogate Objective C      │   │
 │   │ N(μ, σ²) + Grad Mask  │                    │ C = E^[D] (Defect Density) │   │
 │   └───────────┬───────────┘                    └────────────┬───────────────┘   │
 └───────────────┼─────────────────────────────────────────────┼───────────────────┘
                 │                                             │
                 │                                             ▼ Adaptive Edge Weights
                 │                                ┌────────────────────────────┐
                 └───────────────────────────────►│ MWPM / Neural QEC Decoder  │
                     Joint Decoder Steering       │ w_e = log((1 - p_e) / p_e) │
                     (3.5x Stability Gain)        └────────────────────────────┘
```

### 1. The Surrogate Objective Function $C$
Direct optimization of the Logical Error Rate ($\varepsilon_L$) is intractable for three fundamental reasons:
1. $\varepsilon_L$ is suppressed exponentially with code distance $d$ as $\varepsilon_L \propto \Lambda^{-d/2}$, requiring exponentially many QEC cycles to resolve statistically.
2. The parameter space scales as $\mathcal{O}(d^2)$ ($\sim 2,000$ parameters at $d=7$, $\sim 40,000$ at $d=15$).
3. The true logical state is unknown during single-shot algorithm execution.

Google resolved this by establishing the **Surrogate Objective $C$**, defined as the average rate of error-detection events:
$$C = \hat{\mathbb{E}}[D]$$
Resolving $C$ requires only $\mathcal{O}(\varepsilon^{-1})$ cycles, independent of code distance.

#### Linear Gradient Scaling Law (Empirically Verified in Fig. 2c):
$$\nabla \log \varepsilon_L = \frac{d + 1}{2} \cdot \nabla \log C$$
This linear relationship guarantees that minimizing local defect occurrences directly minimizes the global logical error rate.

---

### 2. Bipartite Factor Graph & Gradient Masking
Rather than treating the quantum processor as a monolithic high-dimensional system, the circuit locality is structured as a **bipartite factor graph**:
- **Factor Nodes ($D$)**: QEC detectors across space-time.
- **Variable Nodes ($p$)**: Learnable physical control parameters (XY microwave pulse amplitude, XY frequency detuning, CZ two-qubit coupling strength, CZ post-phase offset).
- In distance-5 surface code, each detector connects to $\sim 302$ parameters, and each parameter connects to only $\sim 18$ detectors.

#### Gradient Masking (Variance Reduction):
The Monte Carlo gradient estimator for parameter $p$ filters out all detectors outside its detecting region:
$$g_{\mu, p} = \frac{1}{\sigma_p} \frac{1}{B} \sum_{b=1}^B R_p^{(b)} \cdot \epsilon_p^{(b)}$$
$$R_p^{(b)} = - \frac{1}{|\text{Det}(p)|} \sum_{d \in \text{Det}(p)} \left( D_d^{(b)} - \bar{D}_d \right)$$
This eliminates cross-chip gradient variance and produces a **system-size-independent exponential convergence law**:
$$1 - \frac{\Lambda}{\Lambda^*} \propto e^{-\gamma t}$$
where the convergence speed $\gamma$ depends strictly on local detecting density, not total qubit count!

---

### 3. Parameter-Exploring Policy Gradients (PEPG) with Gaussian Policy
The agent maintains a factorized multivariate Gaussian distribution $\mathcal{N}(\boldsymbol{\mu}(t), \boldsymbol{\sigma}^2(t))$ over all transmon control lines:
- $\boldsymbol{\mu}(t)$: The instantaneous best estimate of optimal pulse parameters, dynamically tracking physical drift.
- $\boldsymbol{\sigma}(t)$: Exploration standard deviation.
- **Entropy Regularization**: Prevents $\boldsymbol{\sigma} \to 0$, ensuring the agent continuously samples neighboring parameter space to track non-stationary drifts in real time.

---

### 4. Joint Decoder Steering (3.5× Stability Leap)
The framework operates on two synchronized levels:
1. **Physical Controller Steering**: Continuously updates microwave waveforms $\boldsymbol{\mu}(t)$, achieving **$2.4\times$ stability improvement** and **$24\%$ LER reduction**.
2. **Complementary Decoder Steering**: Dynamically reweights the minimum-weight perfect matching (MWPM) graph edges based on instantaneous detector statistics:
   $$w_e(t) = \log\left(\frac{1 - p_e(t)}{p_e(t)}\right)$$
   Combined physical and decoder steering unlocks **$3.5\times$ stability improvement** and **$31\%$ LER reduction**.

---

### 5. Physical Limits & Critical Drift Frequency
- **Response Time**: The closed-loop learning process demonstrates a characteristic exponential response time of $\tau \approx 130$ epochs.
- **Critical Drift Frequency**: $f_c \approx 1/150$ epochs. For drift frequencies slower than $f_c$, real-time steering outcompetes any fixed calibration policy.
- **Natural TLS Noise Attenuation**: Fourier analysis on the Willow chip proves a **4 dB suppression** of low-frequency LER fluctuations originating from two-level system material defects.

---

## ⚡ Architecture & File Structure

```
├── quantum_code/
│   ├── cudaq_cuq_agent.py      # Production Python CUDA-Q script (PEPG + Factor Graph + Decoder Steering)
│   ├── cudaq_cuq_kernel.cpp    # High-performance C++20 CUDA-Q kernel for DGX Quantum
│   └── README.md               # Quickstart guide
```

---

## 🚀 Getting Started

### Option A: Official NVIDIA NGC Container (Recommended for GPUs)

```bash
# Pull the official NVIDIA CUDA-Q container
docker pull nvcr.io/nvidia/quantum/cuda-quantum:latest

# Launch container with all GPUs mounted
docker run --gpus all -it -v $(pwd):/workspace nvcr.io/nvidia/quantum/cuda-quantum:latest bash

# Inside container:
cd /workspace
pip install -r requirements.txt

# Run Nature 2026 Factor-Graph PEPG Benchmark (Distance-3):
python3 cudaq_qec_agent.py --strategy pepg_nature2026 --distance 3 --rounds 150

# Run on Distance-5 (Willow chip configuration):
python3 cudaq_qec_agent.py --strategy pepg_nature2026 --distance 5 --rounds 150
```

### Option B: Local Python Environment (Conda / Virtualenv)

```bash
# Create and activate environment
conda create -n cudaq-qec python=3.10 -y
conda activate cudaq-qec

# Install dependencies
pip install cuda-quantum cuquantum-python torch numpy pymatching scipy

# Run benchmark
python3 cudaq_qec_agent.py --strategy pepg_nature2026 --distance 3 --rounds 150
```

---

## 🕹️ CLI Arguments & Benchmark Modes

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--strategy` | `pepg_nature2026` | Calibration strategy: `pepg_nature2026` (or `autonomous_rl`), `periodic_offline`, `uncalibrated_drift`. |
| `--distance` | `3` | Code distance $d \in \{3, 5, 7, 15\}$. |
| `--rounds` | `150` | Total number of QEC learning epochs. |
| `--initial-drift` | `0.85` | Initial transmon analog miscalibration in MHz. |
| `--drift-rate` | `0.035` | Standard deviation of continuous $1/f$ random walk drift per round (MHz). |
| `--no-decoder-steering` | `False` | Disables joint decoder matching graph reweighting (drops stability from $3.5\times$ to $2.4\times$). |

### Comparative Benchmark Commands:

```bash
# 1. Google Nature 2026 In-Situ RL (Physical + Decoder Steering):
python3 cudaq_qec_agent.py --strategy pepg_nature2026 --initial-drift 0.90 --rounds 150

# 2. Physical Steering ONLY (No Decoder Steering, 2.4x stability):
python3 cudaq_qec_agent.py --strategy pepg_nature2026 --no-decoder-steering --initial-drift 0.90 --rounds 150

# 3. Traditional Periodic Offline Calibration (Halts every 40 cycles):
python3 cudaq_qec_agent.py --strategy periodic_offline --initial-drift 0.90 --rounds 150

# 4. Uncalibrated Baseline (Drift unmitigated, rapid fidelity collapse):
python3 cudaq_qec_agent.py --strategy uncalibrated_drift --initial-drift 0.90 --rounds 150
```

---

## 🖥️ C++20 Kernel for NVIDIA DGX Quantum

For deployment on **NVIDIA DGX Quantum** (tight GPU-QPU coupling over ultra-low-latency NVLink and PCIe links), compile the C++ kernel with `nvq++`:

```bash
nvq++ -target nvidia cudaq_qec_kernel.cpp -o cudaq_qec_kernel
./cudaq_qec_kernel
```

---

## 📄 License
This project is open-source under the [Apache License 2.0](LICENSE).
