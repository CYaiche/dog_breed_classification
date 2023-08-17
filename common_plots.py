
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
    

def display_images(img_list, N, M, from_path=True, titles=None, main_title=None):
    fig, axs = plt.subplots(N,M)
    cnt = 0
    for i in range(N): 
        for j in range(M):  
            img = cv2.imread(img_list[cnt]) if from_path else img_list[cnt]
            if N !=1 : 
                axs[i, j].imshow(cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR))
                axs[i, j].axis("off")
                if titles != None : 
                    axs[i, j].set_title(titles[i+j])
            else : 
                axs[j].imshow(cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR))
                axs[j].axis("off")
                if titles != None : 
                    axs[j].set_title(titles[j])
            cnt = cnt + 1
    if main_title != None : 
        y0 = 0.9 if titles != None else 0.7
        fig.suptitle(main_title,y=y0)
    plt.show()
    
def plot_img(img): 

    plt.axis("off")
    plt.imshow(cv2.cvtColor(img,cv2.IMREAD_ANYCOLOR))
    plt.show()
    
def plot_img_from_path(img_path): 
    assert os.path.exists(img_path) , f"image path does not exist {img_path}"
    imgCV = cv2.imread(img_path)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(imgCV,cv2.IMREAD_ANYCOLOR))
    plt.show()
    
def plot_img_with_rectangles_from_path(img_path, reg_coor):
    assert os.path.exists(img_path) , f"image path does not exist {img_path}"
    imgCV = cv2.imread(img_path)
    
    for cor in reg_coor : 
        pt1  = (int(cor['xmin']), int(cor['ymin']))
        pt2 =  (int(cor['xmax']), int(cor['ymax']))

        imgCV = cv2.rectangle(imgCV, pt1, pt2, (255, 0, 0), 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(imgCV,cv2.IMREAD_ANYCOLOR))
    plt.show()