# Project README

## Overview

This project explores various methods for training and defending against adversarial attacks on deep learning models, specifically using the VGG16 model and the CIFAR10 dataset. The project is divided into several steps, including training the VGG16 model, generating adversarial samples, detecting adversarial samples using steganalysis, and implementing robust defense mechanisms such as region reconstruction and adversarial training (triple network).

## Step 1: Training VGG16

### train.py

- **Model**: Pre-trained VGG16 model.
- **Dataset**: CIFAR10, which should be downloaded and placed in the `datasets/cifar` directory.
- **Data Preprocessing**: Use PyTorch's `transforms` to convert the dataset into Tensor format and `DataLoader` to split the data into batches for training.
- **Checkpoint**: The final model parameters are saved in `ckpt_fin.pkl`, which includes weights, biases, and other necessary parameters.

### classify.py

- **Purpose**: Classify images from the test set using the pre-trained VGG16 model and output the classification accuracy and average loss.
- **Usage**:
  ```bash
  python classify.py -i clean
  ```
- **Results**:
  ```python
  Classification Accuracy: 85.47%
  Average Loss: 0.643
  ```

## Step 2: Generating Adversarial Samples

### attack.py

- **Method**: Fast Gradient Sign Method (FGSM) to generate adversarial samples.
- **Parameters**:
  - `epsilon`: Perturbation magnitude for each pixel, typically between 0 and 1.
- **Process**:
  1. Load the trained model parameters using `torch.load()`.
  2. Generate adversarial samples with FGSM.
  3. Save the original and adversarial images in `.png` format to `sample/adv` and `sample/ori`.
  4. Save the images in `.npy` format to `sample/adv` and `sample/ori`.
  5. Save the labels in `.npy` format to the `sample` directory.
- **Usage**:
  ```bash
  python attack2.py -eps 0.1
  python attack2.py -eps 0.05
  python attack2.py -eps 0.01
  ```

### classify.py

- **Purpose**: Classify adversarial samples using the pre-trained VGG16 model.
- **Usage**:
  ```bash
  python classify.py -i attack
  ```
- **Results**:
  - `eps = 0.01`:
    ```python
    Classification Accuracy: 34.34%
    Average Loss: 5.826
    ```
  - `eps = 0.05`:
    ```python
    Classification Accuracy: 7.11%
    Average Loss: 8.948
    ```
  - `eps = 0.1`:
    ```python
    Classification Accuracy: 7.29%
    Average Loss: 7.096
    ```

## Step 3: Detection of Adversarial Samples Using Steganalysis

### spam.py

- **Purpose**: Extract SPAM features from images using the SPAM algorithm.
- **Functions**:
  - `spam_extract_2`: Extracts SPAM features from an image.
  - `SPAM`: Processes a batch of images to extract SPAM features and returns a feature matrix.
  - `gen_feature`: Generates and saves feature files in the `features` directory.
- **Feature Files**:
  - Clean samples: `clr_f.npy`
  - Adversarial samples: `adv_f.npy`

### fisher.py

- **Purpose**: Train a Fisher Linear Discriminant Analysis (LDA) classifier using SPAM features to detect adversarial samples.
- **Usage**:
  ```bash
  python fisher.py -eps 0.1
  python fisher.py -eps 0.05
  python fisher.py -eps 0.01
  ```
- **Results**:
  - `eps = 0.01`:
    ```python
    Accuracy: 0.724
    ```
  - `eps = 0.05`:
    ```python
    Accuracy: 0.84425
    ```
  - `eps = 0.1`:
    ```python
    Accuracy: 0.91975
    ```

## Step 4: Robust Defense Using Region Reconstruction

### cam_gray_binary.py

- **Purpose**: Generate Class Activation Maps (CAM) using Grad-CAM to visualize the regions of interest in the images.
- **Output**:
  - `ori_{:05d}.png`: Original test image.
  - `cam_gray_{:05d}.png`: Gray-scale CAM image.
  - `cam_binary_{:05d}.png`: Binary CAM image.
  - `cam_jet_{:05d}.png`: Colored CAM image.
  - `stk_{:05d}.png`: Stacked image of the original and colored CAM.
- **Usage**:
  ```bash
  python cam_gray_binary.py
  ```

### fgsm_remove.py

- **Purpose**: Remove the perturbed pixels from adversarial samples using the binary CAM mask.
- **Output**: Modified adversarial samples saved in `fgsm_remove` directory.

### rebuild.py

- **Purpose**: Reconstruct the removed pixels in the adversarial samples using interpolation methods.
- **Output**: Reconstructed images saved in `fgsm_rebuild` directory.
- **Usage**:
  ```bash
  python rebuild.py
  ```

### Final Classification of Rebuilt Images

- **Usage**:
  ```bash
  python classify.py -i rebuild
  ```
- **Results**:
  ```python
  Classification Accuracy: 39.55%
  Average Loss: 3.955
  ```

## Triple Network (Adversarial Training)

1. **Initial Training**:
   - Run `1_train.py` and `2_acc.py` to train the VGG16 model and save the model in `data_model.pth`.
2. **Generate Adversarial Samples**:
   - Use `data_model.pth` to generate adversarial samples and save them in `data_adv`.
3. **Adversarial Training**:
   - Use `data_model.pth` and the adversarial samples from step 2 to train a new model and save it in `data_model_adv.pth`.
4. **Generate Stronger Adversarial Samples**:
   - Use `data_model_adv.pth` to generate stronger adversarial samples and save them in `data_adv_adv`.
5. **Final Training**:
   - Use both the initial adversarial samples and the stronger adversarial samples to train a final model and save it in `data_model_F3`.
6. **Triple Network Decision**:
   - Use the models from steps 1, 3, and 5 to make decisions (voting).

### Results

- **Original Network**:
  - Clean Test Samples: 85.47%
  - Adversarial Test Samples: 15.49%
- **Adversarially Trained Network**:
  - Clean Test Samples: 75.27%
  - Adversarial Test Samples: 95.488%
- **Network Trained on Stronger Adversarial Samples**:
  - Clean Test Samples: 77.88%
  - Adversarial Test Samples: 96.594%
- **Triple Network**:
  - Clean Test Samples: 81.61%
  - Adversarial Test Samples: 94.066%

## Gradio Visualization

- **Usage**:
  ```bash
  python show.py
  ```

## Notes

- For the Chinese version of this README, please refer to `README_zh.md`.

---

Feel free to let me know if you need any further adjustments or additional information!
