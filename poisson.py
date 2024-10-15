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

def shift_image(image, shift_x=0, shift_y=0):
    # Shift the image by shift_x (horizontal) and shift_y (vertical)
    shifted_image = np.roll(image, shift=(shift_y, shift_x), axis=(0, 1))
    
    # Apply 0 padding for the areas that have been shifted out
    if shift_x > 0:
        shifted_image[:, :shift_x] = 0
    elif shift_x < 0:
        shifted_image[:, shift_x:] = 0

    if shift_y > 0:
        shifted_image[:shift_y, :] = 0
    elif shift_y < 0:
        shifted_image[shift_y:, :] = 0
    
    return shifted_image

# Poisson Editing Functions
def compute_laplacian(source):
    """
    Computes the Laplacian of the source image.
    """
    h, w = source.shape
    laplacian = 4*source.copy()
    for shift_x, shift_y in [(-1,0), (1,0), (0,-1), (0,1) ]:
        laplacian -= shift_image(source, shift_x, shift_y)

    return laplacian

def paste_source_borders_on_target(source, target):
    result = target.copy()
    result[[0, -1], :] = source[[0, -1], :]
    result[:, [0, -1]] = source[:, [0, -1]]
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

    A = LinearOperator((N, N), matvec=matvec)
    return A

def lower_(source):
    """
    Computes the Laplacian of the source image.
    """
    result = -(shift_image(source, 0, -1) + shift_image(source, -1, 0))
    result[[0, -1], :] = 0
    result[:, [0, -1]] = 0

    return result


def diag_(source):
    result = source.copy()
    result[1:-1, 1:-1] *= 4
    return result

def diag_inv(source):
    result = source.copy()
    result[1:-1, 1:-1] *= 1/4
    return result

def jacobi_preconditionner(x):
    return diag_inv(x).ravel()

def gauss_seidel_preconditionner(x, w=1):
    rtol = 1e-6
    def nilpotent(x):
        return -w*diag_inv(lower_(x))
    
    result = diag_inv(x.copy())
    nk_of_x = result.copy()
    N = np.prod(x.shape)
    init_norm = np.linalg.norm(result)
    for k in range(1, N):
        nk_of_x = nilpotent(nk_of_x)
        result += nk_of_x
        if False:#np.linalg.norm(nk_of_x) < rtol*init_norm:
            break # should normally go up to N but...

    result /= w
    assert np.allclose(x, w*(diag_(result) + w*lower_(result))), f'the inverse is not correct (adapt the break condition {rtol=:.1E})'

    return result.ravel()


def create_preconditionner(img_shape, w=0):
    """
    Creates a LinearOperator that applies the Laplacian.
    """
    h, w = img_shape
    N = h * w

    if False:# w == 0:
        preconditionner = jacobi_preconditionner
    else:
        preconditionner = lambda x: gauss_seidel_preconditionner(x, 1)

    def matvec(x):
        x_img = x.reshape(h, w)
        return preconditionner(x_img)

    A = LinearOperator((N, N), matvec=matvec)
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
    elif mode == 'alpha_blend':
        result_laplacian = (1-param)*source_laplacian + param*target_laplacian
    elif mode == 'max_blend':
        where_to_take_source = (param*np.abs(source_laplacian) > (1-param)*np.abs(target_laplacian))
        result_laplacian = np.where(where_to_take_source, source_laplacian, target_laplacian)
    else:
        print("Invalid mode selected. Use --help for usage information.")
        quit()
    return result_laplacian

def create_I_minus_Dinv_A(img_shape):
    """
    Creates a LinearOperator that applies the Laplacian.
    """
    h, w = img_shape
    N = h * w

    def matvec(x):
        x_img = x.reshape(h, w)
        laplacian = compute_laplacian(x_img)
        result = paste_source_borders_on_target(x_img, laplacian)
        return x - diag_inv(result).ravel()

    A = LinearOperator((N, N), matvec=matvec)
    return A



def poisson_edit(source, target, mode, param, callback=None):
    """
    Applies Poisson Editing: blends source into target.
    """
    laplacian = construct_laplacian(source, target, mode, param)
    laplacian_with_target_borders = paste_source_borders_on_target(source=target, target=laplacian)

    # Right-hand side of the equation
    b = laplacian_with_target_borders.ravel()

    A = create_operator(source.shape)
    M = create_preconditionner(source.shape)
    I_minus_Dinv_A = create_I_minus_Dinv_A(source.shape)
    import scipy
    from scipy.linalg import interpolative
    rho = interpolative.estimate_spectral_norm(I_minus_Dinv_A)
    print(f'{rho=}')
    exit()

    # Initial guess (zero)
    x = source.copy().ravel() 
    # x = np.zeros_like(b) #rand_like(b)

    # Solve the system using bicgstab
    x, info = bicgstab(A, b, x0=x, maxiter=200, rtol=1e-9, callback=callback)
    callback(x)
    print(f'Done with the optim {info=}')

    return final_blend(target, x)

def final_blend(target, x):
    # Final blended result
    return  x.reshape(target.shape)

class PoissonEditor(object):
    def __init__(self, mode, param):
        self.mode = mode
        self.param = param

    def apply_poisson(self, source, target, callback):
        print('Starting the optimization Loop')

        # Apply Poisson editing with a callback for interactive visualization
        blended_img = poisson_edit(
            source,
            target,
            self.mode,
            self.param,
            callback=callback
        )

        return blended_img


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

class MainWindow():
    def __init__(self, mode, param):
        self.mode = mode
        self.param = param
        self.source_img = (img_as_float(data.text()))
        self.target_img = color.rgb2gray(img_as_float(data.coffee()))
        self.source_selected = False
        self.target_selected = False

        # Create figure and axes

        fig, (ax1, ax2, ax_result) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(self.source_img, cmap='gray')
        ax1.set_title("Source Image")
        ax2.imshow(self.target_img, cmap='gray')
        ax2.set_title("Target Image")
        ax_result.set_title("Blended Image")
        self.ax_result = ax_result

        # Set up region selectors
        source_selector = RegionSelector(ax1, self.source_img, self.on_source_selected)
        target_selector = RegionSelector(ax2, self.target_img, self.on_target_selected)

        # Add Apply Poisson button
        apply_poisson_button = plt.axes([0.4, 0.05, 0.2, 0.075])  # Button placement
        button = plt.Button(apply_poisson_button, 'Apply Poisson Editing')
        button.on_clicked(lambda event: self.on_apply_poisson())

        plt.show()

    def selection_ok(self):
        return self.source_selected and self.target_selected

    def extract_source_region(self, target_shape):
        x1_src, y1_src, x2_src, y2_src = self.source_coords
        source_region = self.source_img[y1_src:y2_src, x1_src:x2_src]
        resized_source = transform.resize(source_region, target_shape, mode='reflect', anti_aliasing=True)
        return resized_source

    def extract_target_region(self):
        x1_tgt, y1_tgt, x2_tgt, y2_tgt = self.target_coords
        target_region = self.target_img[y1_tgt:y2_tgt, x1_tgt:x2_tgt]
        return target_region

    def on_apply_poisson(self):
        if not self.selection_ok():
            return

        target_region = self.extract_target_region()
        source_region = self.extract_source_region(target_region.shape)

        self.iter = 0
        # Create a callback function that updates the blended image
        def callback(x):
            print(f'Running the callback {self.iter=}')
            self.iter += 1
            if self.iter % 10:
                return
            x = x.copy()
            self.blended_img = final_blend(target_region,  x)  # Current solution reshaped as image
            target_region[:, :] = self.blended_img
            self.ax_result.imshow(np.clip(self.target_img,0,1), cmap='gray')
            plt.draw()
            plt.pause(0.01)  # Add a brief pause to allow the plot to update visually
                

        editor = PoissonEditor(self.mode, self.param)
        editor.apply_poisson(source_region, target_region, callback)

        # Display the final result
        self.ax_result.imshow(np.clip(self.target_img,0,1), cmap='gray')
        plt.draw()


    def on_source_selected(self, eclick, erelease):
        self.source_coords = (int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))
        self.source_selected = True
        print(f"Source selected (pixels): {self.source_coords}")

    def on_target_selected(self, eclick, erelease):
        self.target_coords = (int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))
        self.target_selected = True
        print(f"Target selected (pixels): {self.target_coords}")


import argparse

def parse_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Image Editing Tool')

    # Add the '--mode' argument
    parser.add_argument('--mode', type=str, choices=['replace', 'alpha_blend', 'max_blend'], default='max_blend', help='Select editing mode (replace or blend)')

    # Add the '--param' argument
    parser.add_argument('--param', type=float, default=0.2, help='Specify the parameter (transition radius for replace, or blend factor)')

    # Parse the arguments
    args = parser.parse_args()


    return args

def main():
    args = parse_args()

    window = MainWindow(args.mode, args.param)




if __name__ == "__main__":
    main()
