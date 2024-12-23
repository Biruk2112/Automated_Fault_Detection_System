Automated Fault Detection System for Production Lines

Overview

This project demonstrates how to build an automated fault detection system for production lines using Raspberry Pi and Machine Learning. By combining affordable hardware with modern technology, this system captures images of products, trains a Convolutional Neural Network (CNN), and detects defects automatically. It is an accessible solution for small and medium-sized enterprises looking to enhance quality control processes.

Features

Cost-Effective: Uses affordable components like Raspberry Pi and open-source libraries.

Automated Quality Assurance: Simplifies defect detection in production lines.

Customizable: Easily adaptable for various products and defect types.

Scalable: Seamlessly integrates with existing production systems.

Getting Started

Prerequisites

Raspberry Pi (any model with a camera interface).

Camera module or USB camera.

Ultrasonic sensor for product detection.

Optional: Conveyor belt and stepper motor.

Python 3 environment with TensorFlow, NumPy, and Picamera2 installed.

Arduino (optional for motor control).

Installation

Clone the repository:

git clone https://github.com/<username>/automated-fault-detection.git

Navigate to the project directory:

cd automated-fault-detection

Install required Python dependencies:

pip install -r requirements.txt

Usage

Hardware Setup

Set up Raspberry Pi with Raspberry Pi OS.

Connect the camera module and ultrasonic sensor.

Optional: Configure a conveyor belt and lighting for consistent image capture.

Software Setup

Capture images using the provided script:

python images.py

Train the CNN model:

python creation.py

Test the trained model:

python testing_model.py

Deploy the final system:

python final_ver.0.1.py

Project Structure

images.py: Script to capture images for training.

creation.py: Script to create and train the CNN model.

testing_model.py: Script to test the model’s performance.

project_final.ino: Arduino script for motor control and communication.

final_ver.0.1.py: GUI script for final deployment.
