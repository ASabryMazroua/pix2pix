# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:17:13 2022

@author: Ahmed
"""
from pix2pix import Pix2Pix
# from testing_function import random_test


if __name__ == "__main__":    
    gan = Pix2Pix()
    # gan.train(epochs=200, batch_size=1, sample_interval=200)
    gan.train(epochs=3000, batch_size=1, sample_interval=200)

