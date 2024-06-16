# Machine Learning - Neural Network Project - voice classification

Python project . 

Using Pytorch library we trained a model to classify a speech command using speech data (1sec audio word). The provided dataset contains 30 different categories of commands. our task was to train a classifier that classifies this data.

The Gcommand_loader Class enable to create a picture from the audio(.wav) that represent the evolution of the frequency over the time (shade of gray pixels). we used the Google Collab GPU to train our model.

We used Google Colab for this project

## Installation

```bash
pip3 install torch
pip3 install soundfile
```

## Usage

At first:
```python
from google.colab import drive
drive.mount('/content/gdrive')
```
Then:
```python
tar -xvzf ./gdrive/My\ Drive/data.tar.gz
```

We had to create new folder:
```bash
cd ./data
```
```bash
mkdir test_folder
```
```bash
cp -r ./test ./test_folder/
```
```bash
rm -r test
```
```bash
cd ..
```
```bash
python ex4.py
```

## Contributing

This project was elaborated in colaboration between Lior Shimon and Raphael Aben-Moha during Dr Joseph Keshet's class : << Intro to Machine Learning >> at Bar Ilan University - Spring 2019



# complementary:
https://colab.research.google.com/drive/1rImsjjqyEJhaGE4ibtCvXTFq3WD9nap0
data loader tester, ex4.py ,and  gcommand loaderat :  https://drive.google.com/drive/u/0/folders/1RS0OZNH_r0C1nqUdOKOjrC_hWT5Df9hy

