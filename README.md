# Holistic GNNs: Graph Neural Networks Enhanced with Multi-Connectivity Architecture
A novel Graph Neural Network architecture that combines Jumping Knowledge Networks, Virtual Nodes, and Skip Connections for enhanced molecular property prediction and graph-level tasks.

## 🔬 Overview
HolisticGNNs introduces a multi-connectivity approach to Graph Neural Networks by integrating key architectural components:

- Jumping Knowledge Networks: Multi-scale feature aggregation across different network depths
- Virtual Nodes: Global information exchange through auxiliary nodes connected to all graph nodes
- Skip Connections: Residual pathways for improved gradient flow and feature preservation

This combination enables better long-range dependency modeling and enhanced representation learning for complex graph-structured data.

### 🚀 Architectures Included:
- MPNN
- GIN
- GAT
- AttentiveFP
- EEGNN
- TMPNN

## 💻 Environment Installation
#### Run the following commands in the terminal (line by line):
```bash
conda create --name graph python=3.9
conda activate graph
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🏛️ Laboratory of Cheminformatics (LCi)
Faculty of Pharmacy  
Federal University of Goiás (UFG)  
Brazil

👥 **Development Team:**
- Gustavo Felizardo Santos Sandes — [ORCID: 0000-0002-0591-5133](https://orcid.org/0000-0002-0591-5133)  
- Vinícius Alexandre Fiaia Costa — [ORCID: 0000-0001-6479-5963](https://orcid.org/0000-0001-6479-5963)  
- Bruno Junior Neves — [ORCID: 0000-0002-1309-8743](https://orcid.org/0000-0002-1309-8743) 

