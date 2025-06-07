🖼️ Image Captioning using CNN-LSTM
This project implements an image captioning system that generates natural language descriptions for images using deep learning. It combines a Convolutional Neural Network (CNN) for image feature extraction with a Long Short-Term Memory (LSTM) network for sequence modeling and caption generation.

📌 Project Overview
The goal of this project is to train a model that can interpret visual content and generate a human-like caption that describes the image accurately. The system uses:

ResNet (CNN): To extract high-level visual features from the input image.

LSTM (RNN): To generate sequential words based on those features, forming a meaningful sentence.

🧠 Model Architecture
css
Copy
Edit
Image → ResNet (CNN) → Feature Vector → LSTM Decoder → Generated Caption
Encoder: A pre-trained ResNet-50 model is used to encode images into feature vectors.

Decoder: A two-layer LSTM model that receives the feature vector and generates captions word by word.

🗃️ Dataset
The model was trained and tested on a cleaned subset of the Flickr8k dataset. Each image is paired with multiple human-written captions.

🔧 Technologies Used
Python 3

PyTorch

NumPy, Pandas, Matplotlib

NLTK (for text preprocessing)

PIL (for image loading)

torchvision

🚀 How to Run
Clone this repository.

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Prepare your dataset (Flickr8k).

Run the training notebook: imagecaptioning.ipynb

Generate captions using the testing notebook: final result for image captioning.ipynb

📷 Sample Results
Example:

pgsql
Copy
Edit
Input Image: A man riding a wave on a surfboard.
Generated Caption: "a man is surfing on a wave"
📈 Performance
The model was evaluated using BLEU scores, and achieved reasonable accuracy given the simplicity of the architecture and dataset size.

🧑‍💻 Author
Hamza Alaedi
LinkedIn
