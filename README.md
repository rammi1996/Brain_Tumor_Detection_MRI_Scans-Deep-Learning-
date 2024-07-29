
# BRAIN MRI FOR BRAIN TUMOUR DETECTION

## Note
The paths for the datasets in the Python scripts are set according to my system configuration. Adjust the paths in the Python scripts according to your dataset directory before executing. Run the scripts to detect brain tumors from MRI images, using the neural network models specified.

## InceptionV3 ##

- Origin: Developed by researchers at Google, and it's part of the Inception family that started with the original Inception (GoogleNet) model, which was known for winning the ImageNet competition in 2014.

- Structure: InceptionV3 is notable for its complexity and efficiency. It uses a sophisticated architecture composed of "modules" that have several parallel convolutional layers with different filter sizes. This design allows the model to capture information at various scales and complexities within the same level of the network.

- Efficiency: One of the key features of InceptionV3 is its use of 1x1 convolutions to reduce the dimensionality of the data before applying larger convolutions, making it computationally efficient and faster in training without compromising on the depth and breadth of the network.

- Applications: Due to its efficiency and accuracy, InceptionV3 is widely used in many real-world applications involving image classification, object detection, and areas where fine-grained details are crucial.

## VGG16 ##

- Origin: Developed by the Visual Graphics Group (VGG) at the University of Oxford and made a name for itself as a runner-up in the ImageNet competition in 2014.

- Structure: VGG16 is simpler in concept compared to InceptionV3 but is quite deep with 16 layers. It uses a very uniform architecture, exclusively using 3x3 convolutional layers stacked on top of each other in increasing depth. Reducing volume size is handled by max pooling. The uniformity of its architecture allows it to be deeply understood and easy to replicate.

- Performance: While it can be slower and more computationally intensive than more modern architectures due to its depth and the number of parameters, it’s highly regarded for its effectiveness in feature learning and is often used as a base model for various transfer learning tasks.

- Applications: VGG16 has been used extensively in image recognition tasks, especially for those that benefit from deep, hierarchical feature extraction from images.

### GETTING STARTED ###

## Description
This project uses deep learning to detect brain tumors from MRI images. It includes two advanced models, InceptionV3 and VGG16. These models are types of convolutional neural networks, highly effective for image recognition tasks. InceptionV3 is known for its efficiency in handling computationally intensive images, while VGG16 is celebrated for its simplicity and depth.

## Installation
Ensure you have Python installed along with the following libraries, specified with versions to guarantee compatibility:

- **TensorFlow (2.x)**: Essential for building and training the deep learning models.
- **Keras (2.x)**: A high-level neural networks API, used as an interface for TensorFlow.
- **NumPy**: Facilitates extensive operations on large, multi-dimensional arrays and matrices.
- **OpenCV (cv2)**: Applied extensively for image processing tasks.
- **imutils**: Aids in performing basic image processing functions like translation, rotation, resizing, and more, seamlessly with OpenCV.
- **scikit-learn**: Utilized for data splitting into training and test sets and evaluating the models with various metrics.
- **matplotlib**: Useful for plotting graphs and visualizing images.

To install these libraries, execute:
```bash
pip install tensorflow==2.x keras==2.x numpy opencv-python imutils scikit-learn matplotlib
```

## Usage
Run the following scripts to detect brain tumors from MRI images, using the neural network models specified. Modify the scripts to point to your dataset directory before execution.

### InceptionV3 Model
- **File**: `Brain_Tumor_Detection_Inception.py`
  - Processes MRI images using the InceptionV3 model, with preprocessing steps such as resizing and cropping included.
  
  **Command**:
  ```bash
  python Brain_Tumor_Detection_Inception.py
  ```

### VGG16 Model
- **File**: `Brain_Tumor_Detection_model_VGG16.py`
  - Utilizes the VGG16 model to detect tumors, applying similar preprocessing steps as the Inception model.

  **Command**:
  ```bash
  python Brain_Tumor_Detection_model_VGG16.py
  ```

Both scripts require a dataset of MRI images located in a specific directory, which you should configure prior to running the scripts.


## Workflow Overview

The process of detecting brain tumors from MRI images follows these steps:

### Step 1: Data Collection
- Collect brain tumor MRI scans.
- Label each image as "tumor" (yes) or "non-tumor" (no).

### Step 2: Data Augmentation
- Execute the `augment_data.py` script to generate additional images through rotation, flipping, and other transformations, enhancing the dataset size and diversity to prevent overfitting.

### Step 3: Data Loading and Splitting
- Run the `load_data.py` script to load the images and their labels.
- Split the dataset into training (80%), validation (10%), and testing (10%) sets.
- Organize the data into respective folders (e.g., train, val, test).

### Step 4: Model Building and Training
- Use the `model_building.py` script to construct a VGG16-based neural network model.
- Train the model on the training set and save the best model based on validation accuracy.

### Step 5: Model Evaluation
- Employ the `evaluate_model.py` script to assess the trained model on the testing set using metrics like accuracy, precision, recall, and F1 score.

### Step 6: Results Visualization
- Utilize the `plot_metrics.py` script to graphically display training and validation metrics (e.g., loss, accuracy).
- Generate plots and charts to visualize the results.

### Workflow Diagram
Here's a simple diagram illustrating the workflow:

```
                                    +------------------+
                                    | Data Collection  |
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Data Augmentation|
                                    | (augment_data.py)|
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Data Loading and |
                                    | Splitting        |
                                    | (load_data.py)   |
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Model Building   |
                                    | and Training     |
                                    | (model_building.py) |
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Model Evaluation |
                                    | (evaluate_model.py)|
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Results          |
                                    | Visualization    |
                                    | (plot_metrics.py)|
                                    +------------------+
```
This is a simplified diagram and may be adjusted based on specific project needs.



## Workflow Overview for InceptionV3 Model

The process of detecting brain tumors using the InceptionV3 model follows these steps:

### Step 1: Data Collection
- Collect brain tumor MRI scans.
- Label each image as "tumor" (yes) or "non-tumor" (no).

### Step 2: Data Augmentation
- Execute the `augment_data.py` script to generate additional images through techniques like rotation, flipping, and scaling to enhance the dataset size and diversity, which helps in preventing overfitting.

### Step 3: Data Loading and Splitting
- Use the `load_data.py` script to load the images and their corresponding labels.
- Split the dataset into training (80%), validation (10%), and testing (10%) sets.
- Organize the data into respective folders (train, val, test).

### Step 4: Model Building and Training
- Run the `model_building.py` script to build an InceptionV3-based neural network model.
- Train the model on the training data and save the best-performing model based on validation accuracy.

### Step 5: Model Evaluation
- Use the `evaluate_model.py` script to evaluate the model performance on the testing set using metrics such as accuracy, precision, recall, and F1 score.

### Step 6: Results Visualization
- Run the `plot_metrics.py` script to display training and validation metrics such as loss and accuracy.
- Visualize the results through various plots and charts.

### Workflow Diagram
Here's a simplified diagram illustrating the workflow:

```
                                    +------------------+
                                    | Data Collection  |
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Data Augmentation|
                                    | (augment_data.py)|
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Data Loading and |
                                    | Splitting        |
                                    | (load_data.py)   |
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Model Building   |
                                    | and Training     |
                                    | (model_building.py) |
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Model Evaluation |
                                    | (evaluate_model.py)|
                                    +------------------+
                                            |
                                            v
                                    +------------------+
                                    | Results          |
                                    | Visualization    |
                                    | (plot_metrics.py)|
                                    +------------------+
```
This workflow is specifically tailored for the InceptionV3 model and may require adjustments based on your project needs.