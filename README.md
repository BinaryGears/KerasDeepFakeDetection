# KerasDeepFakeDetection Credits
This CNN model is based on this paper: https://doi.org/10.1109/ACCESS.2023.3251417
<br></br>
I have no affiliation with the authors of the paper linked above, I only used this paper as a guideline.
<br></br>
# How to use
<br></br>
Download the latest version of Python 3.11
<br></br>
<br></br>
Run the MakeCSV.py file: ```python MakeCSV.py```
<br></br>
The MakeCSV.py file will make a folder called "images" in the same directory as MakeCSV.py. The "images" folder will have a folder called "test", "train", and "val". Inside the "test" and "val" folders are more folders called "fake_image" and "real_image". Put the real and fake images you have in their proper folders.
<br></br>
Then run MakeCSV.py again: ```python MakeCSV.py```
<br></br>
This will make a .csv file that contains all of the file names and classes of each image, which will be used when you run Main.py later.
<br></br>
<br></br>
TensorFlow recommends upgrading pip before installing TensorFlow.
<br></br>
Use the command: ```pip install --upgrade pip```
<br></br>
<br></br>
This project uses TensorFlow and Keras.
<br></br>
Use the command: ```pip install tensorflow```
<br></br>
<br></br>
You can then run the python file: ```python Main.py```
<br></br>
<br></br>
<br></br>
<br></br>
Link to original repository: https://github.com/BinaryGears/KerasDeepFakeDetection/
