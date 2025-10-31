import cv2
import numpy as np

def manual_low_pass_filter(log_image, kernel_size):
    """
    Manually applies a 2D box blur (averaging filter) to a 2D numpy array .
    This acts as a simple Low-Pass Filter to estimate illumination.
    """
    height, width = log_image.shape
    output = np.zeros_like(log_image)
    
    pad_size = kernel_size // 2
    padded_img = np.pad(log_image, pad_size, mode='reflect')
    
    for y in range(height):
        for x in range(width):
            window = padded_img[y : y + kernel_size, x : x + kernel_size]
            output[y, x] = np.mean(window)
            
    return output

def correct_mono_illumination(image_path):
    """
    Applies homomorphic filtering to a single-channel grayscale image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Convert to log domain
    log_I = np.log(gray_image.astype(float) + 1.0)


    kernel_size = 5 
    
    log_L = manual_low_pass_filter(log_I, kernel_size=kernel_size)
    
    log_R = log_I - log_L
    
    R = np.exp(log_R)
    
    R_normalized = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
    
    cv2.imwrite('reflectance.jpg', R_normalized.astype(np.uint8))
    print("Saved reflectance.jpg")

def correct_color_illumination(image_path):
    """
    Applies homomorphic filtering to a color image while preserving
    true color ratios, as per Part 3 .
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L_channel, a_channel, b_channel = cv2.split(lab_image)
    
    log_L_star = np.log(L_channel.astype(float) / 100.0 + 1.0)

 
    kernel_size = 5

    log_L_est = manual_low_pass_filter(log_L_star, kernel_size=kernel_size)
    log_R_L_star = log_L_star - log_L_est
    R_L_star = np.exp(log_R_L_star)
    
    new_L_channel = cv2.normalize(R_L_star, None, 0, 100, cv2.NORM_MINMAX)
    new_L_channel = new_L_channel.astype(np.uint8)
    
    corrected_lab = cv2.merge([new_L_channel, a_channel, b_channel])
    
    final_image = cv2.cvtColor(corrected_lab, cv2.COLOR_Lab2BGR)
    
    cv2.imwrite('color_reflectance.jpg', final_image)
    print("Saved color_reflectance.jpg ")


if __name__ == "__main__":
    correct_mono_illumination('your_photo.jpg')
    correct_color_illumination('your_photo.jpg')
