<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/JohanHorsmans/cds-language-exam-2021">
    <img src="../README_images/nlp2.png" alt="Logo" width="150" height="150">
  </a>
  
  <h1 align="center">Cultural Data Science 2021</h1> 
  <h3 align="center"><ins>Self Assigned Project:</ins>

Danish hate speech detection</h3> 

  <p align="center">
    Johan Kresten Horsmans & Gustav Aarup Lauridsen
    <br />
    <a href="https://github.com/JohanHorsmans/cds-language-exam-2021/blob/main/Language_Analytics_Exam.pdf"><strong>Link to PDF with all portfolio descriptions »</strong></a>
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#contribution">Contribution</a></li>
    <li><a href="#official-description-from-instructor">Official description from instructor</a></li>
    <li><a href="#methods">Methods</a></li>
    <li><a href="#how-to-run">How to run</a></li>
    <li><a href="#repository-structure-and-contents">Repository structure and contents</a></li>
    <li><a href="#discussion-of-results">Discussion of results</a></li>
  </ol>
</details>

<!-- CONTRIBUTION -->
## Contribution
This project is created in collaboration between [Gustav Aarup Lauridsen](https://github.com/Guscode) and [Johan Kresten Horsmans](https://github.com/JohanHorsmans). Both have contributed equally to every stage of this project from initial conception and implementation, through the production of the final output and structuring of the repository. (50/50%)

<!-- OFFICIAL DESCRIPTION FROM INSTRUCTOR -->
## Project description

### Danish hate speech detection
For our self-assigned project, we wish to see if we can improve the Danish hate-speech detection algorithm that we designed for assignment 5. As stated in assignment 5, we find this task very interesting due to the amount of media coverage on Danish hate speech on social media in recent months. We believe that a robust hate-speech classifier could be a very valuable tool for moderating the tone and retoric of the public debate to make it more constructive.

In assignment 5, we achieved a macro F1-score of 0.71. The current state-of-the-art, as described in [_OffensEval2020_](https://arxiv.org/pdf/2006.07235.pdf), achieves a macro F1-score of 0.81. Our goal with this project is to build a competing state-of-the-art model with similar performance and make it openly available by uploading it, as the first Danish hate speech detection model, to [_huggingface_](https://huggingface.co/). We wish to do this using the [_Nordic Bert_](https://github.com/botxo/nordic_bert)-architecture by [_BotXO_](https://www.botxo.ai/en/blog/danish-bert-model/).

Following this we are going to build a .py-script that can be employed for hate speech classification on ones own dataset. Furthermore, we are creating a Jupyter notebook acting as a tutorial to easily help users deploy the model from huggingface on their own data. Using our huggingface-model will be advantageous, since it takes a long time to train a BERT-model for classification tasks. Using our pretrained model from huggingface, will make it easier and much less time-consuming to implement hate speech moderation for various media-sites and firms who wish to combat Danish hate speech on their online platforms. To improve usability, we will make the model compatible with both a _tensorflow_- and _pytorch_ framework

In summary, the project is comprised of the following steps:
1. Train and test a Nordic Bert-model on the official OffensEval2020-dataset
2. Upload the trained model to huggingface.co
3. Create a Jupyter notebook and .py-script designed to help users deploy the model on their own data.

<!-- METHODS -->
## Methods

__NOTE: Some parts of the following section is repeated from [_assignment 5_](https://github.com/JohanHorsmans/cds-language-exam-2021/tree/main/assignment_5)__

For model training and testing, we are using the OffensEval2020 dataset containing 3000+ Danish comments from Ekstra Bladet and Reddit, labeled with a binary coding scheme indicating offensiveness (link: https://figshare.com/articles/dataset/Danish_Hate_Speech_Abusive_Language_data/12220805).

OffensEval2020 was a competition where researchers and data scientists from all over the world competed to create the best classification models for various languages (including Danish).

The best team in the Danish task achieved a macro F1-score of 0.8119 and the worst team achieved a score of 0.4913. For the full paper, see: [_OffensEval2020_](https://arxiv.org/pdf/2006.07235.pdf)

To make our model-performance comparable to the current state-of-the-art presented in OffensEval2020, we utilized macro F1-score as our evaluation metric:

The F1-score is a metric devised to fuse the relation between model precision and recall into a unified score. The metric is defined as taking the harmonic mean of precision and recall. The reason for using the harmonic mean, rather than the arithmetic mean, is that the harmonic mean of a recall-score of 0 and a precision-score of 100 would result in an F1-score of 0, rather than 50. This is advantageous, since it means that a model cannot achieve a high F1-score by having a high recall or precision by itself. The macro-averaging procedure of the macro F1-score involves calculating the arithmetic mean of the F1-score for each class.

For our modeling, we have chosen to use the Nordic BERT-architecture. The reason behind using Nordic BERT is that it has been deployed with great results in the litterature for a large range of similar classificaion tasks. Furthermore, the winning team in the OffensEval competition for the Danish task also used a Nordic BERT framework.  

We trained the the BERT model for 10-epochs with the following hyperparameters:
* Learning rate: 1e-5,
* Batch size: 16
* Max sequence length: 128

We ran- and developed the code on [_Google Colaboratory_](https://colab.research.google.com/?utm_source=scs-index). For our model-training notebook, please see: "_dk_hate_training.ipynb_"

Our uploaded model can be found here, on huggingface.co: https://huggingface.co/Guscode/DKbert-hatespeech-detection


<!-- HOW TO RUN -->
## How to run

__NOTICE:__ To run the assignment, you need to have configured and activated your virtual environment. See the main [README](https://github.com/JohanHorsmans/cds-language-exam-2021/blob/main/README.md) for a guide on how to do this.

To evaluate the model, please refer to the _dk_hate_detect.py_-script, since this is the main tool. The _dk_hate_detect.ipynb_-notebook is mainly designed as a tutorial for non-expert users. Both contain the same model.

To run the script go through the following steps (__NOTE:__ you have to specify either the _--text_-argument or the _--data_ and _--column_-arguments to run the script):
```bash
cd {root directory (i.e. cds-language-exam-2021)}
cd self_assigned
python3 dk_hate_detect.py
```

__You can specify the following arguments from the terminal:__

_Data path:_
```bash
"--data", 
required = False, 
default = None,
help = "Path to the a dataset in csv format"
```

_Column:_
```bash
"--column"
required = False
default = None,
help = "name of column including text for hatespeech detection "
```

_Single string classification:_
```bash
"--text"
required = False, 
default = None
type = str
help = "string for single string hatespeech detection"
```

_Output:_
```bash
"--output"
required = False
type = str, default = "./"
help = "output path for dataset with hatespeech column"
```

You can also type: ```python3 dk_hate_detect.py -h``` for a detailed guide on how to specify script-arguments. 

Go through the following steps to run the notebook:

Navigate to the "self_assigned"-folder.
Open the "dk_hate_detect.ipynb"-file.
Make sure the kernel is set to _lang_venv_.
You can do this by pressing "kernel" -> "change kernel" -> "lang_venv".

<!-- REPOSITORY STRUCTURE AND CONTENTS -->
## Repository structure and contents

This repository contains the following folder:

|Folder|Description|
|:--------|:-----------|
```data/``` | Folder containing a testing- and training dataset consisting over a 3.000 social media comments labeled after offensiveness (i.e. _NOT_ and _OFF_).

Furthermore, it holds the following files:
|File|Description|
|:--------|:-----------|
```dk_hate_detect.py``` | The python script for the assignment.
```dk_hate_detect.ipynb``` | The Jupyter notebook for the assignment.
```dk_hate_training.ipynb``` | The Jupyter notebook we created when training the model.
```README.md``` | The README file that you are currently reading.



<!-- DISCUSSION OF RESULTS -->
## Discussion of results

Our model achieved a macro F1-score of 0.78. As stated in assignment 5, it is important to note that the dataset is heavily skewed towards non-offensive comments. This skew is also reflected in our model predictions, where the F1-score was much higher for non-offensive comments compared to offensive ones (0.95 vs. 0.60). We believe that this bias towards non-offensive comments, might very well reflect the imbalanced nature of the dataset.

As stated earlier, the currently best performing Danish hate speech model achieved a macro F1-score of 0.81 on the same dataset (as described in OffensEval2020). As such, we have not quite built the new gold-standard model for hate speech detection in Danish. Nonetheless, we have come very close, and our model would have finished 4th place (out of 38 contenders) in the OffensEval2020 competition. Furthermore, it is important to note that we have created the best __publically__ available Danish hate speech + a ready-to-use .py-script and a thorough Jupyter notebook tutorial on how to use it. Therefore, we argue that we have greatly improved the possibilities for an actual real-life implementation of such an algorithm. 

<br />
<p align="center">
  <a href="https://github.com/JohanHorsmans/cds-visual-exam-2021">
    <img src="../README_images/logo_au.png" alt="Logo" width="300" height="102">
  </a>
