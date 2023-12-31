 # Fashion Recommendation System

This project is a fashion recommendation system that allows users to upload images of their clothes and receive outfit suggestions based on the uploaded image. The system uses machine learning models to predict the color and category of the uploaded image, and then generates outfit combinations using other images in the database that match the predicted color and category.

## Prerequisites

To run this project, you will need the following:

* Python 3.6 or later
* TensorFlow 2.0 or later
* Keras 2.3 or later
* Firebase Admin SDK
* Flask
* OpenCV
* NumPy
* Pandas
* Requests
* Pillow
* Rembg

## Installation

To install the required dependencies, run the following command in your terminal:

```
pip install -r requirements.txt
```

## Setup

1. Create a Firebase project and enable the Storage API.
2. Download the Firebase credentials file (`creds.json`) and place it in the `firebase` directory.
3. Create a secret key for JWT authentication and store it in the `SECRET_KEY` environment variable.

## Running the App

To run the app, run the following command in your terminal:

```
flask run
```

The app will run on port 5000 by default.

## Usage

To use the app, follow these steps:

1. Send a POST request to the `/upload` endpoint with the image file as a multipart/form-data field named `image`.
2. The app will process the image, predict its color and category, and generate outfit combinations using other images in the database that match the predicted color and category.
3. The app will return a JSON response with the following information:
    * `message`: A success message indicating that the image was successfully processed and outfit combinations were generated.
    * `url`: The URL of the processed image.
    * `predicted_class`: The predicted category of the image.
    * `color`: The predicted color of the image.

4. Send a GET request to the `/mix-match` endpoint with the `filename` query parameter set to the filename of the image you want to generate outfits for.
5. The app will generate outfit combinations using the selected image and other images in the database that match the predicted color and category of the selected image.
6. The app will return a JSON response with the following information:
     * `message`: Outfit generation completed successfully and uploaded in storage.
    * `status code`: HTTP Status Code of the response
    * `outfit`: Generated outfits from matched filename.
