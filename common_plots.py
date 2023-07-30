
import os 
import cv2 # opencv
import matplotlib.pyplot as plt

def display_color_histogram(imgCV):
    # Get RGB data from image
    blue_color = cv2.calcHist([imgCV], [0], None, [256], [0, 256])
    red_color = cv2.calcHist([imgCV], [1], None, [256], [0, 256])
    green_color = cv2.calcHist([imgCV], [2], None, [256], [0, 256])
    
    # Separate Histograms for each color
    plt.subplot(3, 1, 1)
    plt.title("histogram of Blue")
    plt.hist(blue_color, color="blue")
    
    plt.subplot(3, 1, 2)
    plt.title("histogram of Green")
    plt.hist(green_color, color="green")
    
    plt.subplot(3, 1, 3)
    plt.title("histogram of Red")
    plt.hist(red_color, color="red")
    
    # for clear view
    plt.tight_layout()
    plt.show()
    
    # combined histogram
    plt.title("Histogram of all RGB Colors")
    plt.hist(blue_color, color="blue")
    plt.hist(green_color, color="green")
    plt.hist(red_color, color="red")
    plt.show()