# Deep Learning-Based Signal Sequence Detection with Denoising and Temporal Alignment for Molecular Communication

This repository contains the official implementation of the paper:

**"Deep Learning-Based Signal Sequence Detection with Denoising and Temporal Alignment for Molecular Communication in Vessel-like Channels"**

---

## 📌 Overview

This work proposes a Transformer-based signal sequence detector with a **Denoising and Temporal Alignment Strategy (DTAS)** for molecular communication via diffusion (MCvD) in vessel-like channels with laminar Poiseuille flow.

The repository currently includes:

- ✅ Dataset generation code (MCvD vessel-like simulation framework)
- ✅ Transformer-based detection model implementation

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
- PyTorch 1.12+
- NumPy
- Matplotlib
- Jupyter Notebook

You may install dependencies via:

```bash
pip install torch numpy matplotlib jupyter
```

---

## 🧪 Dataset Generation

The file `generate_dataset.py` implements the vessel-like MCvD simulation framework described in Section II of the paper.

It models:

- Poiseuille flow
- Diffusion-driven propagation
- Multiple transmitters emitting identical molecules
- Receiver absorption process
- Adaptive tail-step extension (G)

To generate dataset:

```bash
python generate_dataset.py
```

The generated dataset contains:

- Received molecular sequence  
- Corresponding transmitted bit matrix  

---

## 🤖 Model Training

The file `training_model.ipynb` contains:

- DTAS preprocessing
- Transformer encoder architecture
- Training & validation pipeline
- Accuracy and BER evaluation

To run training:

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `training_model.ipynb`
3. Run all cells

---

## 📊 Implemented Methods

This repository implements:

- Original (raw sequence)
- Denoised
- Grouped# Deep Learning-Based Signal Sequence Detection with Denoising and Temporal Alignment for Molecular Communication

This repository contains the official implementation of the paper:

**"Deep Learning-Based Signal Sequence Detection with Denoising and Temporal Alignment for Molecular Communication in Vessel-like Channels"**

---

## 📌 Overview

This work proposes a Transformer-based signal sequence detector with a **Denoising and Temporal Alignment Strategy (DTAS)** for molecular communication via diffusion (MCvD) in vessel-like channels with laminar Poiseuille flow.

The repository currently includes:

- ✅ Dataset generation code (MCvD vessel-like simulation framework)
- ✅ Transformer-based detection model implementation

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
- PyTorch 1.12+
- NumPy
- Matplotlib
- Jupyter Notebook

You may install dependencies via:

```bash
pip install torch numpy matplotlib jupyter
```

---

## 🧪 Dataset Generation

The file `generate_dataset.py` implements the vessel-like MCvD simulation framework described in Section II of the paper.

It models:

- Poiseuille flow
- Diffusion-driven propagation
- Multiple transmitters emitting identical molecules
- Receiver absorption process
- Adaptive tail-step extension (G)

To generate dataset:

```bash
python generate_dataset.py
```

The generated dataset contains:

- Received molecular sequence  
- Corresponding transmitted bit matrix  

---

## 🤖 Model Training

The file `training_model.ipynb` contains:

- DTAS preprocessing
- Transformer encoder architecture
- Training & validation pipeline
- Accuracy and BER evaluation

To run training:

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `training_model.ipynb`
3. Run all cells

---

## 📊 Implemented Methods

This repository implements:

- Original (raw sequence)
- Denoised
- Grouped
- DTAS (proposed method)
- DNN baseline (for comparison)

---

## 📖 Citation

If you find this work helpful, please cite our paper:

```
@article{lei2025dtas,
  title={Deep Learning-Based Signal Sequence Detection with Denoising and Temporal Alignment for Molecular Communication in Vessel-like Channels},
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

- This repository contains the core model and dataset generation code.
- You may need to modify the parameter settings in the scripts according to your experimental requirements.
- Please ensure reproducibility by fixing random seeds in your experiments.

---

## 📝 License

This code is released for academic research purposes only.
Please contact the authors for commercial usage.


- DTAS (proposed method)
- DNN baseline (for comparison)

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

- This repository contains the core model and dataset generation code.
- You may need to modify the parameter settings in the scripts according to your experimental requirements.
- Please ensure reproducibility by fixing random seeds in your experiments.

---

## 📝 License

This code is released for academic research purposes only.
Please contact the authors for commercial usage.

