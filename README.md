# anime_face_detection
Members:
- Wai Hong Fung
- Mae Lam
- Ho Kong Leung
- Mayuresh Nerurkar

## Link
Public URL for the page is - `https://github.com/SimonFWH/t413_math_final_project`

## Overview
This project aims to tackle the intriguing challenge of automating the detection of faces in anime images. Unlike traditional face detection, anime characters exhibit a wide range of artistic styles, making this task both captivating and demanding.

## Problem Statement:
Anime characters are portrayed in diverse styles, from highly detailed to minimalistic, and often feature exaggerated facial expressions. This repository addresses the challenge of creating a robust face detection system that can accurately identify and localize faces in this diverse array of artistic representations.

## Data
The dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/andy8744/annotated-anime-faces-dataset), comprises 6640 anime images derived from the top 100 daily rankings on pixiv. The creator of this dataset employed nagadomi's face detector designed for anime/manga using OpenCV. To facilitate its usage, the original dataset underwent a transformation into YOLOv5 PyTorch TXT Annotation Format and was subsequently divided into training, testing, and validation sets in an 80/10/10 ratio using roboflow.

## Scope and Methods
**Custom CNN Model Design and Training Regime**
For tackling the unique challenges of anime face detection, a custom Convolutional Neural Network (CNN) was meticulously crafted from scratch using the PyTorch framework. The design of this model was guided by the distinctive characteristics of anime art styles. 

**Experiments**
Rigorous experiments were executed to validate the effectiveness of the custom CNN model in detecting and localizing anime faces. These experiments not only validate the viability of our approach but also offer insights into potential limitations and avenues for further enhancement. By integrating a tailored CNN architecture, dynamic training visualization, iterative hyperparameter fine-tuning, and dataset refinement, we endeavored to address the intricate task of anime face detection in images.

## Further extensions
Looking ahead, there are several avenues for future exploration and expansion. One approach could involve enhancing the dataset with more diverse annotations, encompassing multiple faces within an image and addressing variations in facial visibility and color schemes. Moreover, exploring more advanced and complex models could yield even better results, especially when dealing with the intricacies of anime images. Additionally, adapting the model for real-time applications or extending it to recognize other elements within anime scenes, such as objects or expressions, could open up new avenues for practical use