# How to use
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
You can then run the python file: ```python TrainReady.py```
<br></br>
<br></br>
<br></br>
<br></br>
# KerasDeepFakeDetection Credits
Paper that this CNN model is in comparison to: https://doi.org/10.1109/ACCESS.2023.3251417
<br></br>
Paper that talks about Separable Convolution Layers: https://arxiv.org/pdf/1610.02357/1000
<br></br>
Paper that talks about PReLU: https://arxiv.org/pdf/1502.01852
<br></br>
Visualkeras citation: @misc{Gavrikov2020VisualKeras,<br></br>
  author = {Gavrikov, Paul},<br></br>
  title = {visualkeras},<br></br>
  year = {2020},<br></br>
  publisher = {GitHub},<br></br>
  journal = {GitHub repository},<br></br>
  howpublished = {\url{https://github.com/paulgavrikov/visualkeras}},<br></br>
}
<br></br>
Link to original repository: https://github.com/BinaryGears/KerasDeepFakeDetection/
