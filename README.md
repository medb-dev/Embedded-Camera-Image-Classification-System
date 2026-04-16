# 📸 Embedded Camera Image Classification System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end production-grade image classification system designed for edge-device scenarios. This project demonstrates the integration of **Deep Learning**, **Asynchronous API design**, and **Clean Architecture** to create a scalable vision service.

---

## 🎯 Project Objective
The goal of this system is to provide a lightweight inference API that can be deployed on low-resource devices (like Raspberry Pi or Jetson Nano) to classify images captured by embedded cameras. It balances the trade-off between model accuracy and inference latency.

## 🚀 Key Features
- **Lightweight CNN**: A custom-built Convolutional Neural Network trained on the CIFAR-10 dataset.
- **Asynchronous API**: Built with FastAPI to handle non-blocking I/O, allowing the system to process multiple image streams simultaneously.
- **OOP Design**: Implemented using a Service-Oriented Architecture (SOA) to ensure separation of concerns between image processing, inference, and transport layers.
- **Robust Validation**: Strict input validation for file types (JPG, PNG) and size limits (10MB) to prevent system crashes.
- **Concurrency Benchmarking**: A dedicated experiment suite to compare Sequential, Threaded, and AsyncIO performance.
- **Automated Pipeline**: Scripts provided for training, evaluation (Confusion Matrix), and live demo.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Deep Learning:** PyTorch, Torchvision
- **API Framework:** FastAPI, Uvicorn
- **Image Processing:** Pillow (PIL), NumPy
- **Evaluation:** Matplotlib, Seaborn, Scikit-learn
- **Testing:** HTTPX, Pandas

---

## 📁 Project Structure
```text
project/
├── app/
│   ├── main.py                 # API Entry point
│   ├── routes/
│   │   └── predict.py          # Prediction endpoints
│   ├── services/
│   │   ├── inference_service.py # Model loading and prediction logic
│   │   └── image_processor.py   # Preprocessing (Resize/Normalize)
│   └── utils/
│       ├── decorators.py        # @log_execution and @timing decorators
│       └── validation.py        # Image file and size validators
├── ml/
│   ├── model.py                # CNN Architecture definition
│   ├── train.py                # Training script
│   ├── evaluate.py             # Accuracy and Confusion Matrix script
│   └── model.pt                # Saved trained weights
├── experiments/
│   └── concurrency_test.py     # Performance benchmarking tool
├── demo/
│   └── demo.py                 # Client-side demonstration script
├── requirements.txt            # Project dependencies
└── Dockerfile                  # Production containerization
