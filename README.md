# Pneumonia-X-Ray-Detector
Artifically intelligent chest x-ray pneumonia detector built in 5 days, using pretrained Model DenseNet.

Overview:
An App to detect Penumonia from Chest X-Rays or validate them as normal. (built in 5 days!)

It can be used on a Batch folder (a folder that contains many xray images of pneumonia or normal or both)

OR

It can be used to detect an xray image

This app was already trained. The folder of model\xray_model.pth contains the training data.

This model can be trained and evaluated.
However, its commented out.
If you would like to train it, follow these steps:
----------------------------------------------------------
- remove commented out code from below # 4 and before # 5
- comment out # 3.5 Code
- Inside the Code.... DATA_DIR = "" #place the ABSOLUTE path of the batch folder inside of data_dir

MAKE SURE THAT the TRAINING folder itself has this structure:

NAMEOFYOURfolder -> train -> NORMAL,PNEUMONIA

NAMEOFYOURfolder -> val -> NORMAL, PNEUMONIA

Where NORMAL is xray pictures of normal xrays

Where PNEUMONIA is xray pictures of pneumonia xrays

Where val is folder to validate predictions and check if its doing good or not.

----------------------------------------------------------

## TO RUN THE APP:

- Install This Xray Trained Model: https://drive.google.com/file/d/1pTYaPE8yO_rNHg-cjC_d-kkyFaSQWfBp/view?usp=drive_link

- Install requirements.txt and run it.


## TO RUN THE APP ON A BATCH FOLDER.
follow these steps:
- Comment out everything below "7.2 Run prediciton App"
- UNCOMMENT Everything Below "7.1 Run prediction batch folder" but BEFORE "7.2 Run prediciton App"
- it would run in the output space of the IDE


Disclaimer

This app is for educational purposes only.
It cannot replace professional medical evaluation.





