# WildFruiP: Estimating Fruit Physicochemical Parameters from Images Captured in the Wild

This is the official implementation of the paper [WildFruiP: Estimating Fruit Physicochemical Parameters from Images Captured in the Wild](https://stillonprogress).

## Abstract

The progress in computer vision has allowed the development of a diversity of precision agriculture systems, improving the efficiency and yield of several processes of farming. Among the different processes, crop monitoring has been extensively studied to decrease the resources consumed and increase the yield, where a myriad of computer vision strategies has been proposed for fruit analysis (e.g., fruit counting) or plant health estimation. Nevertheless, the problem of fruit ripeness estimation
has received little attention, particularly when the fruits are still on the tree. As such, this paper introduces a strategy to estimate the maturation stage of fruits based on images acquired from handheld devices while the fruit is still on the tree. Our approach relies on an image segmentation strategy to crop and align fruit images, which a CNN subsequently processes to extract a compact visual descriptor of the fruit. A non-linear regression model is then used for learning a mapping between descriptors to a set of physicochemical parameters, acting as a proxy of the fruit maturation stage. The proposed method is robust to the variations in position, lighting, and complex backgrounds, being ideal for working in the wild with minimal image acquisition constraints. Source code is available at https://github.com/Diogo365/WildFruiP.

![ProposedMethod](./figures/proposed_method.png)

---

## Data

For downloading our dataset, please follow the [link](https://drive).

## Part 1: Segmentation and alignment of the fruit
  ### Creating the dataset
  
  To create the dataset containing the images and their mask and bounding boxes annotations, execute the following script:
  
    python segmentation.py create-dataset --all_annotations [ALL ANNOTATIONS PATH] --imgs_path [ALL IMAGES PATH]
    
  The result of the execution of the script, should result in the creation of a folder named 'dataset' with the following structure:
      
    ├──WildFruiP
    |  └── dataset
    │      ├── test
    │      ├── train
    │      └── valid
  
  ### Training
  
  To train the segmentation/detection model, execute the following script:
  
    python segmentation.py train
  
  ### Inference
  
  To infer/predict the segmentation mask of an image, execute the following script:
  
    python segmentation.py inference --img [IMAGE PATH] --model [MODEL PATH] --type [TYPE]
  
  Type is either 'fig' or 'prickly_pear'.
  
  ### Testing alignment in an image
  
  To align an image you can use the following command:
  
    python segmentation.py align --img [IMAGE PATH] --model [MODEL PATH] --type [TYPE]
  
  Type is either 'fig' or 'prickly_pear'.
  
  ### Prepare dataset for feature extraction
  
  To use the images for the second part, they need to be all aligned in the image. To performe that action, please execute the following command:
  
    python segmentation.py align-folder --source_dir [SOURCE DIR] --desti_dir [DESTINATION DIR] --model [MODEL PATH] --type [TYPE]

## Part 2: Estimating Fruit Physicochemical Parameters

### Explanation

After the creation of the folder with all images aligned, the new dataset is created. To perform the train of the model it will be needed the Excel. The Excel contains all the parameter values of each fruit species and it is available [here](https://excel).

### Training

To train the model for estimating the fruit physicochemical parameters, please execute the following command:

    python estimate_parameters.py train --img_path [IMAGE PATH] --excel_path [EXCEL PATH] --type [TYPE]

This will start the train of all parameters, using the K-fold cross-validation technique (it may take a while to finish the train of all parameters).

### Evaluating the Model

To evaluate the model's performance of each parameter, we measure the mean performance of the folds. To perform this action, please execute the following command:

    python estimate_parameters.py eval-model --parameter [PARAMETER] --img_path [IMAGE PATH] --excel_path [EXCEL PATH] --type [TYPE] --model [MODEL FOLDER] 

### Inference

To estimate the fruit physicochemical parameters it is needed to supply the max and min used in the train of the model. The following command is used to perform inference:

    python estimate_parameters.py inference --model [MODEL PATH] --img [IMAGE PATH] --type [TYPE] --max [MAX] --min [MIN]
    
## Citation

If you find this work useful in your research, please consider citing:

```
@proceedings{paulo2023wildfruip,
  author = {Diogo J. Paulo and João C. Neves and Cláudia M. B. Neves and Dulcineia Ferreira Wessel},
  title = {WildFruiP: Estimating Fruit Physicochemical Parameters from Images Captured in the Wild},
  booktitle = {26th Iberoamerican Congress on Pattern Recognition (CIARP)},
  year = {2023}
}
```

## Acknowledgements
We would like to express our heartfelt gratitude to the [InovFarmer.MED](https://inovfarmer-med.org/pt-pt) project, which is part of the
PRIMA Programme, for their invaluable support and collaboration throughout the course of our research. We acknowledge the financial support from [InovFarmer.MED](https://inovfarmer-med.org/pt-pt) and [NOVA LINCS (UIDB/04516/2020)](https://nova-lincs.di.fct.unl.pt/), which enabled us to conduct field experiments, gather data and analyze results. The success of this scientific endeavor would not have been possible without their dedicated support.
