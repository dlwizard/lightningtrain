import unittest
from PIL import Image
import requests
from io import BytesIO

import logging
LOGGER = logging.getLogger(__name__)

import time

class TestCifar10Model(unittest.TestCase):
    def test_vit_prediction(self):
        # Load an image from your local filesystem
        image_path = 'images/airplane.jpg'
        pil_image = Image.open(image_path)
        image_bytes = BytesIO()
        pil_image.save(image_bytes, format='JPEG')

        # Send a GET request to the FastAPI app
        url = 'http://192.168.0.103:8080/infer_vit'
        files = [('image', ('image.jpg', image_bytes.getvalue()))]
        start = time.time()
        for _ in range(100):
            response = requests.get(url, files=files)
        
        end = time.time()
        time_taken = end - start

        LOGGER.info(f"Average time taken for each VIT inferencing: {time_taken / 100}s")

        # Validate the response
        self.assertEqual(response.status_code, 200, "Request failed with status code: {}".format(response.status_code))
        
        result = response.json()
        self.assertEqual(len(list(result.keys())), 10, "Invalid response received")

    def test_gpt_prediction(self):
        # Send a GET request to the FastAPI app
        url = 'http://192.168.0.103:8080/infer_gpt/'

        start = time.time()
        for _ in range(100):
            response = requests.get(url, json={'sentense': 'Hello harry potter'})
        
        end = time.time()
        time_taken = end - start
        
        LOGGER.info(f"Average time taken for predicting 256 GPT tokens: {time_taken / 100}s")

        # Validate the response
        self.assertEqual(response.status_code, 200, "Request failed with status code: {}".format(response.status_code))

        result = response.json()
        self.assertEqual(result['sentense'], 'Hello harry potter', "Invalid response received")

        self.assertTrue(len(list(result.keys())) > 1, "Invalid response received")

        self.assertTrue(len(result['completion'])>1, "Invalid tokens received")



if __name__ == '__main__':
    unittest.main()