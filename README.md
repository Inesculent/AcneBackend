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

Segmentation with efficientnet-b1 backbone on approx ~400 images. We use dice coeff with binary segmentation to classify the acne "cells" on our dataset.

Dataset carefully curated from variety of Kaggle Datasets, annotated by hand.
https://www.kaggle.com/datasets/inesculent/acne-larger-set/data

Please reference the [frontend](https://github.com/Inesculent/AcneFrontend) for additional information.





