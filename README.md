IFT780-TP3
Link to the Best Model
Our best model is available on Google Drive at the following address: Google Drive Link.

Dataset
For this third practical assignment, you are required to perform segmentation on the Automated Cardiac Diagnosis Challenge (ACDC) dataset1. Start by executing:

bash
Copy code
pip install -r requirements.txt
The classes are as follows:

0: background,
1: Right ventricle,
2: Myocardium,
3: Left ventricle.
The data contains only one modality (cine-MRI) and is acquired in 3D.

Provided Code

The provided code allows you to train a model for N epochs, specifying the learning rate, optimizer, and batch size. You are free to modify the code as needed to meet the assignment requirements.

To Do:
The following sections describe the features you need to add to the code for managing your training processes. Note: All these features must be modifiable via the command line!

Architectures (3 points)
You must implement three different architectures (excluding the one provided) and design one of your own. If you lack inspiration, you can take cues from the IFT780Net used in TP2. Your report should include a figure representing your custom architecture, as seen in the course notes. For the other two, it is recommended to research well-performing segmentation architectures from the internet or course materials.

Hyperparameter Search (2 points)
You must modify the provided code to perform a hyperparameter search for the implemented models. Considerable hyperparameters might include the learning rate, optimizer, or specific parameters of your architectures.

Checkpointing (1 point)
You must implement a method to save the model during training and resume it after an interruption (e.g., a power outage or an accidental Ctrl-C).

Data Augmentation (2 points)
You must add functionalities to apply various types of data augmentation during training.

Bonus: Alternative Loss Function (1 point)
While cross-entropy is currently used as the loss function, many alternatives exist. You must implement another loss function for training the network, explain its advantages, and demonstrate them. For inspiration: 2.

Report
For each of the features mentioned above, you must specify in your report how and where in the code they were added, along with examples of their usage and their impact on training. You must also describe the process used to find the best combination of training parameters that yield the best performance on the test set.

Finally, include relevant learning curves and an analysis of them. Your report should be comprehensive enough to allow me to easily reproduce your experiments.

Code (2 points)
You will be evaluated on the quality and usability of the submitted code. Imagine you are giving this code to a colleague who hasn’t taken this course but would still like to train neural networks. I should not need to modify the code during grading. It should be possible to change the model, hyperparameters, perform a hyperparameter search, etc., by modifying the command-line arguments only. Nothing should be hardcoded.

Submission
You must submit your report, code, and the weights of your best model so that I can test its performance.

Footnotes
ACDC Dataset ↩

Jadon, S. (2020, October). A survey of loss functions for semantic segmentation. In 2020 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB) (pp. 1-7). IEEE. arXiv Link ↩
