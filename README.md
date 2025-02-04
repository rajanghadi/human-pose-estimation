# human-pose-estimation

This repository contains a project that detects and analyzes human poses using machine learning techniques. It identifies key body points and visualizes them as a stick figure, making it useful for applications like fitness tracking, sports analysis, and interactive systems. This project implements human pose estimation using AI/ML techniques with both TensorFlow and OpenVINO backends. It leverages Python and core libraries for implementation and visualization.

![Human pose estimation output showing detected keypoints on a person.](public/images/human_pose_estimation_example_1.png)

## Overview

Human pose estimation is a computer vision task that aims to identify the location of key body joints (e.g., elbows, knees, wrists) in an image or video. This project provides implementations using two different frameworks:

*   **TensorFlow:** A popular open-source machine learning framework.
*   **OpenVINO:** Intel's toolkit for optimizing and deploying deep learning models.

## Technologies Used

This project utilizes the following technologies:

*   **Programming Language:** Python
*   **Machine Learning Frameworks:** TensorFlow, PyTorch, OpenVINO
*   **Core Libraries:** Streamlit, OpenCV-Python, Pillow, NumPy

## Implementations

The project includes implementations for human pose estimation using:

*   TensorFlow
*   OpenVINO

## Requirements

*   **Operating System:** Ubuntu
*   **Python:** Python 3.8
*   **PyTorch:**

## Prerequisites

1.  **Dataset:** This project uses the COCO dataset. Download it from [http://cocodataset.org/](http://cocodataset.org/).

2.  **Install Requirements:**

    ```bash
    pip install streamlit opencv-python pillow numpy tensorflow torch torchvision torchaudio
    ```

## Getting Started

1.  **Project Setup:** Copy the project files to a directory of your choice. Create `data` and `models` directories.

2.  **Run the Streamlit app:**

    ```bash
    streamlit run your_streamlit_app.py
    ```

3.  **Running Inference:**

    ```bash
    python run_tensorflow_inference.py --model_path ./models/your_tensorflow_model.h5 --image_path ./images/test_image.jpg
    python run_openvino_inference.py --model_path ./models/your_openvino_model.xml --image_path ./images/test_image.jpg
    ```
