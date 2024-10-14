import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage import data, img_as_float, color, transform

def combine_gradually(a, b, max_distance_fraction):
    # Check if images are the same size
    if a.shape != b.shape:
        raise ValueError("Input images must be of the same size.")
    h, w = a.shape
    max_distance = max_distance_fraction * min(h, w)
    # Create a meshgrid for calculating distances from the border
    y_indices, x_indices = np.indices((h, w))
    # Calculate the distance from the nearest border
    distance_from_border = np.minimum(np.minimum(x_indices, w - 1 - x_indices),
                                      np.minimum(y_indices, h - 1 - y_indices))
    # Create the gradation image
    gradation_image = np.clip(distance_from_border / max_distance, 0, 1)
    return a*gradation_image + b*(1-gradation_image)

# Poisson Editing Functions
def compute_laplacian(source):
    """
    Computes the Laplacian of the source image.
    """
    h, w = source.shape
    laplacian = np.zeros_like(source)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Compute Laplacian for interior pixels
            laplacian[i, j] = (
                4 * source[i, j]
                - source[i + 1, j]
                - source[i - 1, j]
                - source[i, j + 1]
                - source[i, j - 1]
            )

    return laplacian

def paste_source_borders_on_target(source, target):
    h, w = source.shape
    result = target.copy()
    for i in range(1, h - 1):
        for j in [0, w-1]:
            result[i, j] = source[i, j]
    for j in range(0, w):
        for i in [0, h-1]:
            result[i, j] = source[i, j]
    return result


def create_operator(img_shape):
    """
    Creates a LinearOperator that applies the Laplacian.
    """
    h, w = img_shape
    N = h * w

    def matvec(x):
        x_img = x.reshape(h, w)
        laplacian = compute_laplacian(x_img)
        result = paste_source_borders_on_target(x_img, laplacian)
        return result.ravel()

    def rmatvec(x):
        return matvec(x)

    A = LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)
    return A


def rand_like(x):
    return np.random.rand(*x.shape)


def construct_laplacian(source, target, mode, param):
    source_laplacian = compute_laplacian(source)
    target_laplacian = compute_laplacian(target)
    if mode == 'replace':
        result_laplacian = source_laplacian
        if param > 0:
            result_laplacian = combine_gradually(source_laplacian, target_laplacian, param)
    elif mode == 'blending':
        result_laplacian = (1-param)*source_laplacian + param*target_laplacian
    else:
        print("Invalid mode selected. Use --help for usage information.")
        quit()
    return result_laplacian

def poisson_edit(source, target, mode, param, callback=None):
    """
    Applies Poisson Editing: blends source into target.
    """
    laplacian = construct_laplacian(source, target, mode, param)
    laplacian_with_target_borders = paste_source_borders_on_target(source=target, target=laplacian)

    # Right-hand side of the equation
    b = laplacian_with_target_borders.ravel()

    A = create_operator(source.shape)

    # Initial guess (zero)
    x = np.zeros_like(b) #rand_like(b)

    # Solve the system using bicgstab
    x, info = bicgstab(A, b, x0=x, maxiter=200, callback=callback)
    print(f'Done with the optim {info=}')

    return final_blend(target, x)

def final_blend(target, x):
    # Final blended result
    return  x.reshape(target.shape)

# GUI Setup

class RegionSelector:
    def __init__(self, ax, img, callback):
        self.ax = ax
        self.img = img
        self.callback = callback
        self.selector = RectangleSelector(
            self.ax, self.onselect, useblit=True,
            button=[1],  # Left click only
            interactive=True
        )
    
    def onselect(self, eclick, erelease):
        # Store pixel coordinates
        self.callback(eclick, erelease)

def on_source_selected(eclick, erelease):
    global source_selected, source_coords
    source_coords = (int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))
    source_selected = True
    print(f"Source selected (pixels): {source_coords}")

def on_target_selected(eclick, erelease):
    global target_selected, target_coords
    target_coords = (int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))
    target_selected = True
    print(f"Target selected (pixels): {target_coords}")


class PoissonEditor(object):
    def __init__(self, mode, param):
        self.source_img = color.rgb2gray(img_as_float(data.astronaut()))
        self.target_img = color.rgb2gray(img_as_float(data.coffee()))
        self.mode = mode
        self.param = param

    def on_apply_poisson(self, ax_result):
        if source_selected and target_selected:
            # Extract the source region using NumPy views
            x1_src, y1_src, x2_src, y2_src = source_coords
            source_region = self.source_img[y1_src:y2_src, x1_src:x2_src]

            # Target region
            x1_tgt, y1_tgt, x2_tgt, y2_tgt = target_coords
            target_region_for_display = self.target_img[y1_tgt:y2_tgt, x1_tgt:x2_tgt]
            target_region = target_region_for_display.copy()

            # Resize the source region to match the target region size
            resized_source = transform.resize(source_region, target_region.shape, mode='reflect', anti_aliasing=True)

            # Create a callback function that updates the blended image
            def callback(x):
                blended_img = final_blend(target_region,  x)  # Current solution reshaped as image
                target_region_for_display[:,:] = blended_img
                ax_result.imshow(np.clip(self.target_img,0,1), cmap='gray')
                plt.draw()
                plt.pause(0.01)  # Add a brief pause to allow the plot to update visually
                

            # Apply Poisson editing with a callback for interactive visualization
            blended_img = poisson_edit(
                resized_source,
                target_region,
                self.mode,
                self.param,
                callback=callback
            )

            # Display the final result
            target_region_for_display[:,:] = blended_img
            ax_result.imshow(np.clip(self.target_img,0,1), cmap='gray')
            plt.draw()

import argparse

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Image Editing Tool')

    # Add the '--mode' argument
    parser.add_argument('--mode', type=str, choices=['replace', 'blending'], help='Select editing mode (poisson or blending)')

    # Add the '--param' argument
    parser.add_argument('--param', type=float, default=0.2, help='Specify the parameter (transition radius for replace, or blend factor)')

    # Parse the arguments
    args = parser.parse_args()


    return args

def main():
    args = parse_args()

    editor = PoissonEditor(args.mode, args.param)

    source_selected = False
    target_selected = False

    # Create figure and axes
    fig, (ax1, ax2, ax_result) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(editor.source_img, cmap='gray')
    ax1.set_title("Source Image")
    ax2.imshow(editor.target_img, cmap='gray')
    ax2.set_title("Target Image")
    ax_result.set_title("Blended Image")

    # Set up region selectors
    source_selector = RegionSelector(ax1, editor.source_img, on_source_selected)
    target_selector = RegionSelector(ax2, editor.target_img, on_target_selected)

    # Add Apply Poisson button
    apply_poisson_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # Button placement
    button = plt.Button(apply_poisson_button, 'Apply Poisson Editing')
    button.on_clicked(lambda event: editor.on_apply_poisson(ax_result))

    plt.show()



if __name__ == "__main__":
    main()
