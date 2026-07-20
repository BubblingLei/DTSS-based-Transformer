# Deep Learning-Based Signal Detection with Denoising and Temporal Sequence Structuring for Molecular Communication in Vessel-like Channels

This repository contains the official implementation of the paper:

**"Deep Learning-Based Signal Detection with Denoising and Temporal Sequence Structuring for Molecular Communication in Vessel-like Channels"**

---

## 📌 Overview

This work proposes a Transformer-based signal sequence detector with a **Denoising and Temporal Sequence Structuring (DTSS) strategy** for molecular communication via diffusion (MCvD) in vessel-like channels with laminar Poiseuille flow. The proposed approach jointly addresses long-range inter-symbol interference (ISI) and inter-link interference (ILI) under indistinguishable molecular signatures from multiple transmitters.

The repository currently includes:

- ✅ Dataset generation code (MCvD vessel-like simulation framework)
- ✅ Transformer-based detection model with DTSS preprocessing

---

## 📂 Repository Structure

```
.
├── generate_dataset.py     # Vessel-like MCvD dataset generation
├── training_model.ipynb    # Model training and evaluation notebook
├── README.md
```

---

## ⚙️ Environment Requirements

Recommended environment:

- Python 3.9+
- PyTorch 2.0+
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

You may install dependencies via:

```bash
pip install torch numpy pandas matplotlib jupyter
```

---

## 🧪 Dataset Generation

The file `generate_dataset.py` implements the vessel-like MCvD simulation framework described in Section II of the paper.

It models:

- Three-dimensional Poiseuille flow in a cylindrical vessel
- Diffusion-driven molecular propagation
- Multiple transmitters (TNs) emitting identical molecules
- Receiver (RN) absorption and vessel wall absorption
- Adaptive tail-step extension ($G$) for capturing delayed arrivals

Default parameters:
- Vessel radius $R_{\mathrm{ves}} = 5~\mu\mathrm{m}$
- Receiver radius $R_{\mathrm{RN}} = 1~\mu\mathrm{m}$
- Number of transmitters $M = 4$
- Diffusion coefficient $D = 120~\mu\mathrm{m}^2/\mathrm{s}$
- Mean flow velocity $v_f = 300~\mu\mathrm{m}/\mathrm{s}$
- Number of released molecules per bit 1: $N_0 = 10000$

To generate dataset:

```bash
python generate_dataset.py
```

The generated dataset contains the received molecular count sequences and corresponding transmitted bit matrices, which are saved as CSV files ready for model training.

---

## 🤖 Model Training

The file `training_model.ipynb` contains the full pipeline:

- **DTSS preprocessing**: Denoising (head/tail trimming with $G/2$ configuration $C_3$) and temporal sequence structuring (partitioning into $L$ consecutive segments of $K_{\mathrm{b}}$ steps)
- **Transformer encoder architecture**: Multi-head self-attention with learnable positional embeddings and FFN with GELU activation
- **Training & validation pipeline**: AdamW optimizer with weight decay, linear learning rate warm-up, ReduceLROnPlateau scheduler, early stopping
- **Accuracy and BER evaluation**: Bit-level metrics over all $M \times L$ transmitted bits per sample

To run training:

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `training_model.ipynb`
3. Run all cells sequentially

The notebook performs an ablation study comparing six strategies:

| Strategy | Model Architecture | Input Description |
|----------|-------------------|-------------------|
| DNN | Original DNN | Raw sequences covering the full simulation window |
| DTSS-DNN | DNN with DTSS | Trimmed + temporally structured sequences |
| Transformer | Original Transformer | Raw sequences covering the full simulation window |
| Denoised | Transformer with Denoised strategy | Trimmed sequences only |
| Grouped | Transformer with Grouped strategy | Raw sequences partitioned into $L$ segments |
| DTSS-Transformer | Transformer with DTSS | Trimmed + temporally structured sequences |

---

## 📊 Key Results

Under the baseline parameters ($L=10$, $K_{\mathrm{b}}=18$, $v_f=300~\mu\mathrm{m}/\mathrm{s}$, $d_x=15~\mu\mathrm{m}$):

- **DTSS-Transformer** achieves the highest test accuracy of **94.25%**
- The Grouped strategy outperforms the original Transformer by **11.17%**
- DTSS-Transformer outperforms DTSS-DNN by **9.60%**, demonstrating the synergy between DTSS and self-attention
- DTSS-Transformer exhibits only **0.17%** accuracy degradation when increasing sequence length $L$ from 5 to 40, compared to 22.00% for the original Transformer
- DTSS-Transformer maintains accuracy above **99%** even at extended transmitter–receiver distances (up to $20~\mu\mathrm{m}$)

---

## 📖 Citation

If you find this work helpful, please cite our paper:

```
@article{lei2025dtss,
  title={Deep Learning-Based Signal Detection with Denoising and Temporal Sequence Structuring for Molecular Communication in Vessel-like Channels},
  author={Lei, Yuqing and Cheng, Zhen and Xia, Ming and Yao, Xinwei and Li, Qu},
  journal={IEEE Transactions on ...},
  year={2025}
}
```

(Please update citation details after publication.)

---

## 📬 Contact

For questions regarding the paper or the code, please contact:

**First Author:**  
Yuqing Lei  
Email: 302023315176@zjut.edu.cn  

School of Computer Science and Technology  
Zhejiang University of Technology  
Hangzhou, China

---

## 🔒 Notes

- This repository contains the core model and dataset generation code
- You may need to modify the parameter settings in the scripts according to your experimental requirements
- Please ensure reproducibility by fixing random seeds in your experiments
- The simulation framework is designed for Linux environments with Python 3.10

---

## 📝 License

This code is released for academic research purposes only.
Please contact the authors for commercial usage.
