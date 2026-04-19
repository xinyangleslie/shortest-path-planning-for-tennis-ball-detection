Tennis Ball Click Labeler

==================================================
1. Purpose
==================================================
This tool is used to label tennis balls by mouse clicking.
It generates small fixed bounding boxes around each clicked point
and exports the annotations in YOLO format.

Main idea:
- Open one image at a time
- Left click on each tennis ball center
- Press 'd' to generate fixed boxes
- Press 's' to save labels and move to the next image
- The program automatically exports a .txt file in YOLO format

==================================================
2. Required Environment
==================================================
Python 3.x

Required Python packages:
- opencv-python

Install OpenCV if needed:
pip install opencv-python

==================================================
3. File Preparation
==================================================
Put all images that need annotation into one folder.

Supported image formats:
- .jpg
- .jpeg
- .png
- .bmp
- .webp

Example:
images/
    img001.jpg
    img002.jpg
    img003.png

==================================================
4. How to Run
==================================================
Open terminal or command prompt in the folder containing:
- tennis_click_labeler.py

Run:

python tennis_click_labeler.py --image_dir ./images --output_dir ./fixed_box_output --class_id 0

Parameter explanation:
--image_dir
    Folder containing input images

--output_dir
    Folder for saving labels and visualization results

--class_id
    YOLO class id
    For tennis ball, usually use 0

Example on Windows:
python tennis_click_labeler.py --image_dir "C:\Users\YourName\Desktop\images" --output_dir "C:\Users\YourName\Desktop\fixed_box_output" --class_id 0

==================================================
5. Mouse and Keyboard Operations
==================================================
Mouse:
- Left click:
    Add one red point on a tennis ball
- Right click:
    Remove the most recently added point

Keyboard:
- d
    Generate bounding boxes from all clicked points
- s
    Save current labels and move to the next image
    Note: press 'd' first before saving
- u
    Undo the last point
    If already in review mode, it returns to click mode first
- r
    Reset current image annotations
- n
    Skip current image without saving
- q
    Quit the program

==================================================
6. Recommended Labeling Workflow
==================================================
For each image:

Step 1:
Open the image in the program

Step 2:
Left click each tennis ball once
Try to click near the center of the ball

Step 3:
If one point is wrong, use:
- right click, or
- press 'u'

Step 4:
After all points are added, press:
d

This will generate small fixed boxes

Step 5:
Check whether the generated boxes look reasonable

Step 6:
If everything looks correct, press:
s

This saves the YOLO label file and moves to the next image

==================================================
7. Output Files
==================================================
The program creates two folders inside output_dir:

1) labels/
    Contains YOLO annotation text files
    Example:
    img001.txt

2) vis/
    Contains saved visualization images
    These images show the clicked points and generated boxes
    Example:
    img001.jpg

==================================================
8. YOLO Label Format
==================================================
Each line in the saved .txt file is:

class_id x_center y_center width height

All values are normalized to [0, 1].

Example:
0 0.512500 0.635417 0.010938 0.014583

==================================================
9. Box Size Logic
==================================================
This tool does NOT detect objects automatically.
It creates fixed-size boxes based on mouse clicks.

Box rule:
- For most image areas:
    use a small box
- For points in the bottom 1/4 of the image:
    use a slightly larger box

Reason:
Objects near the bottom may appear a little larger in the image.

==================================================
10. Important Notes
==================================================
1. Please click only once per tennis ball
2. Try to click near the center of each ball
3. Press 'd' before pressing 's'
4. If you skip an image by pressing 'n', no label file will be saved
5. If you close with 'q', the program stops immediately
6. This tool is designed for fast manual annotation, not automatic detection

==================================================
11. Simple Explanation of the Code Logic
==================================================
The code works in the following way:

- It reads all images from the input folder
- It displays one image at a time
- Each left mouse click records one point
- The program converts each point into a fixed-size bounding box
- The boxes are then transformed into YOLO format
- The labels and preview images are saved to the output folder

So the key idea is:
mouse click -> point -> fixed box -> YOLO label file

==================================================
12. Troubleshooting
==================================================
Problem:
No images found

Possible reason:
The image folder path is wrong, or the folder is empty

Problem:
Program says "Please press 'd' first."

Reason:
You tried to save before generating boxes

Problem:
OpenCV window does not appear

Possible reason:
OpenCV is not installed correctly, or the Python environment is not the one you expected

==================================================
13. Suggested User Reminder
==================================================
When labeling:
- Click each tennis ball center once
- Press 'd' to generate boxes
- Check the boxes
- Press 's' to save
- Use right click or 'u' if one point is wrong
