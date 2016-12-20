# KINE
Course project for EECS E4750 Hybrid Computing, Fall 2016, Columbia University

Our project implement a hole filling algorithm followed by a bilateral filter.

The .py file KINE.py is the main code of this project.
PyCUDA is required to run this code.
The serial part takes quiet a long time to run comparing to the parallel part, you maybe want to comment the serial part if you just want to check the functionality.
The code is specialized for the test demo (depthImage.txt or Depth.jpg, you want to put it in the directory where the code is running), but with all the comments, you can easily modify the code and test it with other depth image.

The folder named results includes the results of running the code with the test demo.
