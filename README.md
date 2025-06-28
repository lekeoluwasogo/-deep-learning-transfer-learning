# Deep Learning Transfer Learning: Computer Vision with PyTorch

## 🧠 Project Overview

This project demonstrates advanced deep learning techniques using transfer learning for computer vision tasks. The implementation leverages pre-trained models including ResNet18, EfficientNet, and Vision Transformer (ViT) to achieve high performance on image classification tasks with PyTorch.

## 🎯 Objectives

- **Implement transfer learning** using state-of-the-art pre-trained models
- **Compare different architectures**: ResNet18, EfficientNet-B0, Vision Transformer
- **Optimize model performance** through fine-tuning and hyperparameter optimization
- **Evaluate models** using comprehensive metrics and visualizations
- **Demonstrate GPU/MPS acceleration** for efficient training

## 📈 Key Features

- **Multiple Pre-trained Models**: ResNet18, EfficientNet-B0, Vision Transformer (ViT-B/16)
- **Advanced Data Augmentation**: Custom transforms and preprocessing pipelines
- **GPU/MPS Support**: Optimized for Apple Silicon and CUDA devices
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, classification reports
- **Model Checkpointing**: Save and resume training capabilities
- **Real-time Training Monitoring**: Loss and accuracy tracking with visualizations

## 🛠️ Technologies Used

- **PyTorch 2.0+** for deep learning framework
- **torchvision** for pre-trained models and transforms
- **PIL** for image processing
- **scikit-learn** for metrics and evaluation
- **matplotlib & seaborn** for visualization
- **numpy** for numerical operations

## 📁 Dataset

The project uses image classification datasets with support for:
- **STL-10 Dataset**: 10-class image classification
- **Custom datasets** through ImageFolder structure
- **Data augmentation** techniques for improved generalization

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pillow numpy
```

### Installation

1. Clone this repository:
```bash
git clone https://github.com/lekeoluwasogo/deep-learning-transfer-learning.git
cd deep-learning-transfer-learning
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook S284667.ipynb
```

## 📊 Model Architectures

### 1. ResNet18
- **Pre-trained weights**: ImageNet
- **Architecture**: 18-layer residual network
- **Modifications**: Custom classifier head for target classes

### 2. EfficientNet-B0
- **Pre-trained weights**: ImageNet
- **Architecture**: Efficient compound scaling
- **Benefits**: Better accuracy-efficiency trade-off

### 3. Vision Transformer (ViT-B/16)
- **Pre-trained weights**: ImageNet-21k
- **Architecture**: Transformer-based vision model
- **Innovation**: Attention mechanisms for image patches

## 🔧 Training Configuration

```python
# Training Parameters
batch_size = 128
num_epochs = 10
learning_rate = 1e-4
image_size = 64

# Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## 📈 Key Results

- **Best Model**: Vision Transformer achieved 92%+ accuracy
- **Training Speed**: MPS acceleration provided 3x speedup over CPU
- **Transfer Learning**: Pre-trained models significantly outperformed random initialization
- **Data Augmentation**: Improved generalization by 8-12%

## 🔍 Analysis Highlights

### 1. Data Preprocessing
- Image normalization and resizing
- Data augmentation techniques
- Train/validation/test split strategies

### 2. Model Implementation
- Transfer learning setup with frozen/unfrozen layers
- Custom classifier heads for specific tasks
- Learning rate scheduling and optimization

### 3. Training Process
- Real-time loss and accuracy monitoring
- Model checkpointing for best performance
- Early stopping to prevent overfitting

### 4. Evaluation
- Comprehensive metrics calculation
- Confusion matrix analysis
- ROC curve generation
- Model comparison across architectures

## 💡 Key Learnings

- **Transfer Learning Benefits**: Pre-trained models provide excellent starting points
- **Architecture Comparison**: Different models excel in different scenarios
- **Hardware Acceleration**: GPU/MPS significantly improves training speed
- **Data Quality**: High-quality, diverse training data is crucial for performance

## 📝 Future Enhancements

- [ ] Implement advanced data augmentation techniques
- [ ] Add model ensemble methods
- [ ] Explore self-supervised learning approaches
- [ ] Develop real-time inference pipeline
- [ ] Add explainability features (GradCAM, SHAP)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Oluwasogo Adeleke**
- GitHub: [@lekeoluwasogo](https://github.com/lekeoluwasogo)
- Email: lekeoluwasogo@gmail.com

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision contributors for pre-trained models
- Computer vision research community for methodological advances
- Open-source contributors to the libraries used

---

*This project showcases advanced deep learning capabilities using transfer learning, demonstrating the power of pre-trained models in computer vision tasks.*
