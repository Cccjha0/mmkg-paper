# Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion

[![Paper](https://img.shields.io/badge/Paper-ACL2026-blue.svg)]((https://arxiv.org/abs/2509.23714))
[![Code](https://img.shields.io/badge/Code-GitHub-green.svg)](https://github.com/zjukg/M-Hyper)

This is the official repository for the paper: **"Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion"**, accepted at **ACL 2026 (Main Conference)**.

## 🌟 Abstract

Multi-modal knowledge graph completion (MMKGC) aims to discover missing facts in multi-modal knowledge graphs (MMKGs) by leveraging both structural relationships and diverse modality information of entities. Existing MMKGC methods follow two multi-modal paradigms: fusion-based and ensemble-based. Fusion-based methods employ fixed fusion strategies, which inevitably leads to the loss of modality-specific information and a lack of flexibility to adapt to varying modality relevance across contexts. In contrast, ensemble-based methods retain modality independence through dedicated sub-models but struggle to capture the nuanced, context-dependent semantic interplay between modalities. To overcome these dual limitations, we propose a novel MMKGC method **M-Hyper**, which achieves the coexistence and collaboration of fused and independent modality representations. Our method integrates the strengths of both paradigms, enabling effective cross-modal interactions while maintaining modality-specific information. Inspired by "quaternion" algebra, we utilize its four orthogonal bases to represent multiple independent modalities and employ the Hamilton product to efficiently model pair-wise interactions among them. Specifically, we introduce a Fine-grained Entity Representation Factorization (FERF) module and a Robust Relation-aware Modality Fusion (R2MF) module to obtain robust representations for three independent modalities and one fused modality. The resulting four modality representations are then mapped to the four orthogonal bases of a biquaternion for comprehensive modality interaction. Extensive experiments indicate its state-of-the-art performance with better robustness.


## 🚀 Quick Start

### Dependencies
- Python==3.9
- numpy==1.24.2
- scikit_learn==1.2.2
- torch==2.0.0
- tqdm==4.64.1
- Maybe other library version also works.

### Data Download
You should first download the pre-trained multi-modal embeddings from [Google Drive](https://drive.google.com/drive/folders/191u4WhT_7P9Ze8Q9N3U3rdr8IWEKYMry?usp=sharing) and put them in the `datasets/{each dataset}/` path.

### Train and Evaluation
```shell script
nohup python run.py --dataset MKG-W --model M_Hyper_B --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer wN3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save > ../log/M_Hyper-MKG-W.log &
nohup python run.py --dataset MKG-Y --model M_Hyper_B --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer wN3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save > ../log/M_Hyper-MKG-Y.log &
nohup python run.py --dataset DB15K --model M_Hyper_B --rank 128 --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer wN3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save > ../log/M_Hyper_DB15K.log &
```

## 🖊️ Citation
If you find our work useful, please consider citing:
```bibtex
@article{liu2025collaboration,
  title={Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion},
  author={Liu, Zhiqiang and Zhang, Yichi and Sun, Mengshu and Liang, Lei and Zhang, Wen},
  journal={arXiv preprint arXiv:2509.23714},
  year={2025}
}
```
