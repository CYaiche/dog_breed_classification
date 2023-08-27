
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
    
def plot_train_val_accuracy_and_loss(history, show=True):
    acc      = history.history['accuracy']
    val_acc  = history.history['val_accuracy']
    loss     = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(loss))

    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    if show : 
        plt.show()
    else :
        return fig 

def display_images(img_list, N, M, from_path=True, titles=None, main_title=None, correct_color=False):
    fig, axs = plt.subplots(N,M, figsize=(2*M,2*N))
    cnt = 0
    for i in range(N): 
        for j in range(M):  
            img = cv2.imread(img_list[cnt]) if from_path else img_list[cnt]
            if correct_color :
                img = cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR)
            if N !=1 : 
                axs[i, j].imshow(img)
                axs[i, j].axis("off")
                if titles != None : 
                    axs[i, j].set_title(titles[cnt])
            else : 
                axs[j].imshow(img)
                axs[j].axis("off")
                if titles != None : 
                    axs[j].set_title(titles[cnt])
            cnt = cnt + 1
    if main_title != None : 
        y0 = 1 
        fig.suptitle(main_title,y=y0)
    plt.show()
    
def plot_img(img, correct_color=False): 
    if correct_color :
        img = cv2.cvtColor(img,cv2.IMREAD_ANYCOLOR)
    plt.axis("off")
    plt.imshow(img)
    plt.show()
    
def plot_img_from_path(img_path, correct_color=False): 
    assert os.path.exists(img_path) , f"image path does not exist {img_path}"
    imgCV = cv2.imread(img_path)
    if correct_color :
        imgCV = cv2.cvtColor(imgCV,cv2.IMREAD_ANYCOLOR)
    plt.axis("off")
    
    plt.imshow(imgCV)
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