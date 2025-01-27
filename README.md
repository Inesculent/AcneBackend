Backend for AcneSegmentation
---------------------------------

Video Demo
----------------
[![Acne Detection](https://img.youtube.com/vi/eicBLMGiN2c/0.jpg)](https://www.youtube.com/watch?v=eicBLMGiN2c)



**To run**
-----------------

1. Install requirements *pip install requirements.txt*
2. Ensure that you have a .env file with the proper information for your database.
3. Use the .sql files in /sq to initialize your database tables.
4. Run the application via *uvicorn main:app*



**Overview**
-----------------

Acne is a common skin condition that affects approximately 85% of people between the ages of 12 and 24, according to the Acmerican Academy of Dermatology.
It is characterized by inflamed or infected sebacous glands, commonly presenting itself as red, swollen pimples on the face, back, and chest, though other areas can be affected.

When looking at existing ML models for Acne, I realized that there were an abudance of detection models, but very few — if any — that focused on segmentation. In addition, the datasets that I was able to find had very poor labeling, with many false negatives, leading to any model trained off them to be plauged with underfitting issues. In addition, even though detection based models were able to accurately pinpoint acne some of the time, they could not paint a clear picture on the scale and severity of the acne, as the "number" of acne cells only tells part of the picture.

As a result, I resolved to create my own segmentation dataset, and train a CNN on it. The largest issue I faced in this was in fine tuning a model that was capable of picking up numerous small features, while also avoiding false positives on other skin conditions, such as rosacea or scarring.

Segmentation with efficientnet-b1 backbone on approx ~400 images. We use dice coeff with binary segmentation to classify the acne "cells" on our dataset.

Dataset carefully curated from variety of Kaggle Datasets, annotated by hand.
https://www.kaggle.com/datasets/inesculent/acne-larger-set/data

Please reference the [frontend](https://github.com/Inesculent/AcneFrontend) for additional information.





