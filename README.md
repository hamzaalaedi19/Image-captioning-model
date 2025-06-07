
#  Image Captioning using CNN-LSTM

This project implements an image translation system that generates natural language Arabic descriptions using deep learning. This system combines a **convolutional neural network (CNN)** to extract image features and a **long-short-term memory (LSTM)** network to model sequences and generate translations.

---

##  Project Overview

The goal of this project is to train a model that can interpret visual content and generate a human-like caption that describes the image accurately. The system uses:

- **ResNet (CNN)**: To extract high-level visual features from the input image.
- **LSTM (RNN)**: To generate sequential words based on those features, forming a meaningful sentence.

---

##  Model Architecture

```
Image → ResNet (CNN) → Feature Vector → LSTM Decoder → Generated Caption
```

- **Encoder**: A pre-trained ResNet-50 model is used to encode images into feature vectors.
- **Decoder**: A two-layer LSTM model that receives the feature vector and generates captions word by word.

---

##  Dataset

The model was trained and tested on a cleaned subset of the **Flickr8k** dataset. Each image is paired with multiple human-written captions.

---

##  Technologies Used

- Python 3
- Tensorflow
- Keras
- PyTorch
- Transformers
- NumPy, Pandas, Matplotlib
- NLTK (for text preprocessing)
- PIL (for image loading)
- torchvision

---

##  How to Run

1. Clone this repository.
2. Prepare your dataset (Flickr8k).
3. Run the training notebook: `imagecaptioning.ipynb`
4. Generate captions using the testing notebook: `final result for image captioning.ipynb`


---

##  Performance

The model was evaluated using **BLEU scores**, and achieved reasonable accuracy given the simplicity of the architecture and dataset size.

---

##  Author

Hamza Alaedi  
[LinkedIn](https://www.linkedin.com/in/hamza-alaedi-395669366)  


