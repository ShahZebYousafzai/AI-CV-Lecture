import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(img, title="Image", cmap=None):
    """Display a single OpenCV image (BGR or grayscale) using Matplotlib."""
    plt.figure(figsize=(6, 5))
    if len(img.shape) == 3:
        # Colour image: convert BGR → RGB for Matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        # Grayscale image
        plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def show_multiple(images, titles, cols=3, figsize=(14, 4)):
    """Display multiple images side by side."""
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap="gray")
        axes[i].set_title(title, fontsize=10)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()

print("Helper functions defined.")