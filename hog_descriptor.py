import cv2
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib

from functools import partial
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
from skimage import feature


# matplotlib.use("tkagg")
# Global variables.
pressed = False
rect = Rectangle((0, 0), 0, 0, facecolor="None", edgecolor="green", linestyle="dashed")
x0 = 0
y0 = 0
width = 0
height = 0
roi = np.zeros((1, 1), np.uint8)
pixels_per_cell = 8
cells_per_block = 2
bins = 8


def onPress(event):
    """
    Callback to handle the mouse being clicked and held over the canvas.
    """

    global pressed, x0, y0, width, height

    # Check the mouse press was actually on the canvas.
    if event.xdata is not None and event.ydata is not None:
        # Upon initial press of the mouse record the origin and record the mouse as pressed.
        pressed = True
        x0 = int(event.xdata)
        y0 = int(event.ydata)
        width = height = 0


def on_release(event):
    """
    Callback to handle the mouse being released over the canvas.
    """
    global pressed, roi

    # Check that the mouse was actually pressed on the canvas to begin with and this isn't a rouge mouse
    # release event that started somewhere else.
    if pressed:

        # Upon release draw the rectangle as a solid rectangle.
        pressed = False

        # Check if the user has only pressed in a single points on the image.
        if width == 0 or height == 0:
            return

        # Get the rectangle coordinates.
        (x, y) = rect.get_xy()
        w = rect.get_width()
        h = rect.get_height()

        # Region of Interest.
        roi = image[y : y + h, x : x + w].copy()
        roi_ax.imshow(roi)
        roi_ax.set_title("Region of Interest (%d X %d)" % (roi.shape[1], roi.shape[0]))
        plt.draw()

        # Compute HOG.
        compute_hog()


def on_motion(event):
    """
    Callback to handle the motion event created by the mouse moving over the canvas.
    """

    global rect, width, height

    # If the mouse has been pressed draw an updated rectangle when the mouse is moved so
    # the user can see what the current selection is
    if pressed:

        # Check the mouse was released on the canvas, and if it wasn't then just leave the width and
        # height as the last values set by the motion event
        if event.xdata is None and event.ydata is None:
            return

        # Calculate the width and height of bouding box.
        width = int(event.xdata) - x0
        height = int(event.ydata) - y0

        # Aspect ratio.
        if width < height:
            width = int(height * 0.5)
        else:
            height = int(width * 0.5)

        # Set the width and height and draw the rectangle
        rect.set_width(width)
        rect.set_height(height)
        rect.set_xy((x0, y0))
        plt.draw()


def update_ppc(slider, val):
    """
    This function will be performed when the user changes the pixel per cell slider.
    """
    global pixels_per_cell, cells_per_block

    # Create a mask for the slider to set only 2^i.
    slider.val = int(round(val))
    slider.poly.xy[2] = slider.val, 1
    slider.poly.xy[3] = slider.val, 0
    slider.valtext.set_text(slider.valfmt % (np.power(2, slider.val)))

    # Change the number of pixels per cell.
    pixels_per_cell = np.power(2, slider.val)

    # Validate the cells per block.
    if cells_per_block >= (64 / pixels_per_cell):
        cells_per_block = int(64 / pixels_per_cell) - 1
        update_cpb(cpb_ax, cells_per_block)

    # Compute HOG.
    compute_hog()


def update_cpb(slider, val):
    """
    This function will be performed when the user changes the cells per block slider.
    """
    global cells_per_block

    # Validate the cells per block.
    val = int(round(val))
    if val >= (64 / pixels_per_cell):
        val = int(64 / pixels_per_cell) - 1

    # Create a mask for the slider to set only 2^i.
    slider.val = int(round(val))
    slider.poly.xy[2] = slider.val, 1
    slider.poly.xy[3] = slider.val, 0
    slider.valtext.set_text(slider.valfmt % slider.val)

    # Change the number of pixels per cell.
    cells_per_block = slider.val

    # Compute HOG.
    compute_hog()


def update_orientation(slider, val):
    """
    This function will be performed when the user changes the orientation slider.
    """
    global bins

    # Create a mask for the slider to set only 2^i.
    slider.val = int(round(val))
    slider.poly.xy[2] = slider.val, 1
    slider.poly.xy[3] = slider.val, 0
    slider.valtext.set_text(slider.valfmt % slider.val)

    # Change the number of bins.
    bins = slider.val

    # Compute HOG.
    compute_hog()


def compute_histogram(magnitude, orientation, bins=8, isPlotting=False):
    """
    This function computes the polar histogram.
    """
    # Number of bins in the histogram.
    orientation = orientation / 360 * 2 * np.pi
    bins_number = bins
    thetas = np.linspace(0.0, 2 * np.pi, bins_number + 1)

    # Compute the histogram based on the orientation and magnitude.
    hist, _, _ = hist_ax.hist(
        orientation.ravel(), thetas, weights=magnitude.ravel(), density=True
    )
    hist_ax.cla()

    width = 2 * np.pi / bins_number

    # Plot the polar histogram.
    if isPlotting:
        bars = hist_ax.bar(
            thetas[:bins_number],
            hist,
            width=width,
            bottom=0.0,
            color=".8",
            edgecolor="k",
        )
        for bar in bars:
            bar.set_alpha(0.5)
        plt.draw()

    return hist


def compute_gradients(image):
    """
    This function computes the orientation and magnitude of input image.
    """
    # Check if the input image is grayscale.
    if len(image.shape) == 2:
        grayscale = image.copy()
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Magnitude and Orientation images.
    magnitude = np.zeros(grayscale.shape)
    orientation = np.zeros(grayscale.shape)

    # Calculate the gradients.
    grad_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    magnitude, orientation = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    print("\n\nbins:\n", bins)
    print("\n\n\n****************\n\n\n\n")

    # Plot the polar histogram.
    compute_histogram(magnitude, orientation, bins, True)

    return magnitude, orientation


def compute_hog():
    """
    Compute the Histogram of Oriented Gradients.
    """
    # Check if the user has selected the Region of Interest.
    if roi.shape[0] == 1:
        return

    # Resized image.
    resized = cv2.resize(roi, (64, 128))

    # Sobel and HOG images.
    sobel = hog = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Compute the gradient.
    sobel, orientation = compute_gradients(sobel)

    (H, hog) = feature.hog(
        hog,
        orientations=bins,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        block_norm="L2",
        visualize=True,
        feature_vector=True,
    )

    # Draw pixels per cell.
    for i in range(0, resized.shape[0], pixels_per_cell):
        cv2.line(resized, (0, i), (63, i), (0, 255, 0))

    for j in range(0, resized.shape[0], pixels_per_cell):
        cv2.line(resized, (j, 0), (j, 127), (0, 255, 0))

    # Draw the cells per block.
    k = pixels_per_cell * cells_per_block
    cv2.rectangle(resized, (0, 0), (k, k), (0, 0, 255))

    plt.ioff()
    # Resized image.
    plt.pause(0.0001)
    resized_ax.imshow(resized)
    resized_ax.set_title("Resized (%d X %d)" % (resized.shape[1], resized.shape[0]))

    # Sobel image.
    plt.pause(0.0001)
    sobel_ax.imshow(sobel, cmap="gray")
    sobel_ax.set_title("Sobel (%d X %d)" % (sobel.shape[1], sobel.shape[0]))

    # HOG image.
    plt.pause(0.0001)
    hog_ax.imshow(hog, cmap="gray")
    hog_ax.set_title("HOG (%d X %d)" % (hog.shape[1], hog.shape[0]))

    plt.pause(0.0001)
    plt.draw()
    plt.ion()


# Input image.
filename = "./inputs/people01.jpg"
image = cv2.imread(filename)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Matplotlib grid.
G = gridspec.GridSpec(2, 4)

# Input image subplot.
image_ax = plt.subplot(G[0, 0:2])
image_ax.add_patch(rect)
image_ax.imshow(image)
image_ax.set_title("Input Image (%d X %d)" % (image.shape[1], image.shape[0]))
image_ax.set_axis_off()

# Connect the mouse events to their relevant callbacks
plt.connect("button_press_event", onPress)
plt.connect("button_release_event", on_release)
plt.connect("motion_notify_event", on_motion)

# Histogram subplot.
hist_ax = plt.subplot(G[0, 2:4], polar=True)
hist_ax.set_title("Polar Histogram")
hist_ax.set_xticklabels([])
hist_ax.set_yticklabels([])

# Region of Interest subplot.
roi_ax = plt.subplot(G[1, 0])
roi_ax.set_title("Region of Interest")
roi_ax.set_axis_off()

# Resized subplot.
resized_ax = plt.subplot(G[1, 1])
resized_ax.set_title("Resized")
resized_ax.set_axis_off()

# Sobel subplot.
sobel_ax = plt.subplot(G[1, 2])
sobel_ax.set_title("Sobel")
sobel_ax.set_axis_off()

# HOG subplot.
hog_ax = plt.subplot(G[1, 3])
hog_ax.set_title("HOG")
hog_ax.set_axis_off()

# Define the pixel per cell slider.
axcolor = "lightgoldenrodyellow"
ppc_ax = plt.axes([0.1225, 0.02, 0.78, 0.03], facecolor=axcolor)
ppc_ax = Slider(ppc_ax, "Pixel per Cell", 1, 5, valinit=3, valfmt="%i")
ppc_ax.on_changed(partial(update_ppc, ppc_ax))
ppc_ax.valtext.set_text("8")

# Define the cells per block slider.
axcolor = "lightgoldenrodyellow"
cpb_ax = plt.axes([0.1225, 0.06, 0.78, 0.03], facecolor=axcolor)
cpb_ax = Slider(cpb_ax, "Cells per Block", 2, 7, valinit=2, valfmt="%i")
cpb_ax.on_changed(partial(update_cpb, cpb_ax))
cpb_ax.valtext.set_text("2")

# Define the orientation slider.
axcolor = "lightgoldenrodyellow"
orientation_ax = plt.axes([0.1225, 0.1, 0.78, 0.03], facecolor=axcolor)
orientation_ax = Slider(orientation_ax, "Orientation", 2, 20, valinit=8, valfmt="%i")
orientation_ax.on_changed(partial(update_orientation, orientation_ax))
orientation_ax.valtext.set_text("8")

plt.show()
