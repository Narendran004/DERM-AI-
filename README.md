[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Intel_logo_%282020%2C_light_blue%29.svg/300px-Intel_logo_%282020%2C_light_blue%29.svg.png" width="50">](https://www.intel.com/)
[<img src="https://www.intel.com/content/dam/develop/public/us/en/images/admin/oneapi-logo-rev-4x3-rwd.png" width="50">](https://www.intel.com/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter%20Notebook-%23F37626.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23000.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)

# DermAI: Skin Disease Prediction and Solutions

DermAI is a machine learning project developed to predict skin diseases based on input images and provide solutions for further action. This project utilizes a convolutional neural network (CNN) trained on a dataset of skin disease images to make predictions.

# Demonstration of the Project

https://github.com/Vijay18003/DermAi/assets/158248736/fdcd7c7b-b452-425a-a137-3139bc5119e7



 
# Tech Stacks Used

- Python
- TensorFlow/Keras
- Streamlit
- Jupter Notebook

# What DermAI does:

- Image Analysis: DermAI accepts input images of skin lesions or conditions, captured through various means such as smartphone cameras or digital cameras.

- Predictive Modeling: The input images are processed through a pre-trained convolutional neural network (CNN) model. This model has been specifically trained on a diverse dataset of skin disease images to recognize patterns and features indicative of different skin conditions.

- Disease Identification: Based on the analysis performed by the CNN model, DermAI identifies the potential skin disease or condition present in the input image. This can include a wide range of dermatological issues such as acne, eczema, psoriasis, melanoma, etc.

- Solution Recommendations: After identifying the skin disease or condition, DermAI provides recommendations for further action. This may include suggesting over-the-counter treatments, advising a visit to a dermatologist for further evaluation, or providing general skincare advice.

- Continuous Learning: DermAI may incorporate mechanisms for continuous learning and improvement. This could involve updating the model with new data to enhance its accuracy and expand its capabilities over time.

Overall, DermAI aims to assist individuals in early detection and management of skin diseases by leveraging the power of machine learning and image recognition technology. By providing timely and accurate predictions, it helps users make informed decisions about their skin health and seek appropriate medical attention when necessary.
# how DermAI works:

- Data Collection and Preprocessing: DermAI starts by collecting a diverse dataset of skin disease images. This dataset includes images of various skin conditions such as acne, eczema, psoriasis, melanoma, etc. These images are labeled with their corresponding skin diseases for supervised learning. Before training the model, the images may undergo preprocessing steps like resizing, normalization, and augmentation to enhance the robustness and generalization of the model.

- Training the Convolutional Neural Network (CNN): DermAI utilizes a convolutional neural network (CNN), a type of deep learning model well-suited for image classification tasks. The CNN is trained on the dataset of skin disease images using techniques like backpropagation and gradient descent to optimize its parameters. During training, the CNN learns to extract relevant features from the input images and map them to specific skin diseases.

- Model Evaluation: After training, DermAI evaluates the performance of the CNN using a separate validation dataset. This evaluation helps ensure that the model generalizes well to unseen data and accurately predicts skin diseases.

- Prediction: Once trained and validated, DermAI is ready to make predictions on new input images of skin lesions or conditions. Users can upload images of their skin issues through an interface provided by DermAI.

- Feature Extraction and Classification: The input images are processed through the trained CNN. The CNN extracts features from the images at various levels of abstraction through convolutional layers. These features are then fed into fully connected layers for classification. DermAI predicts the most likely skin disease or condition based on the features extracted from the input image.

- Solution Recommendation: Based on the predicted skin disease, DermAI provides recommendations for further action. This may include suggesting over-the-counter treatments, advising a visit to a dermatologist for further evaluation, or providing general skincare advice. The recommendations aim to assist individuals in managing their skin health effectively.

- Continuous Improvement: DermAI may incorporate mechanisms for continuous learning and improvement. This could involve updating the model with new data, fine-tuning its parameters, or incorporating feedback from users and dermatologists to enhance its accuracy and relevance over time.

Overall, DermAI leverages the power of machine learning and image recognition technology to assist individuals in early detection and management of skin diseases. By providing timely and accurate predictions, along with actionable recommendations, DermAI aims to empower users to take proactive steps towards maintaining their skin health.

# Intel One Api
Intel One Api had a great impact on our project by utilizing the services provided by them we were able to significantly reduce the execution and training time 
of our ML model `cimta`.
![d8fa6c72-f88a-4111-bd30-b40698151150](https://github.com/t-aswath/mdeditor/assets/119417646/01dbaa20-3499-4c71-bc01-c48b71ae2b79)
![536dabf3-68d6-4d07-abc0-199e3362521d](https://github.com/t-aswath/mdeditor/assets/119417646/904aa562-7b50-4a1a-9c2b-5f9f08be1a5f)

# References For Dataset
-The dataset used for training the model is available on  [![Kaggle](https://img.shields.io/badge/Kaggle-%23000.svg?style=flat&logo=Kaggle&logoColor=white)](https://www.kaggle.com/)
