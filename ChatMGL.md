# ChatMGL: A Large Language Model Fine-tuned for Data Science Questions
Manos Chatzakis (emmanouil.chatzakis@epfl.ch), Ioannis Bantzis (ioannis.bantzis@epfl.ch), Lluka Stojollari (lluka.stojollari@epfl.ch)

## Getting started
To start, install all needed python packages. You need to support the python version(s) listed in "python.txt".
```sh
pip install -U -r requirements.txt
```
Our implementation should work with any Python version >= 3.7, but we have tested our work only with the listings in "python.txt".

## Model Download
The final ChatMGL model has a size of >3GBs, thus it cannot be uploaded to this GitHub repo. However, we provide a google drive link containing ChatMGL:
```sh
https://drive.google.com/drive/folders/1Klcx6gJHiJIj-BS6vB2-OM9Gx49qyxPd
```

We also provide a UNIX script that downloads the dataset inside /models/chatMGL/ directory. To run it, use:
```sh
chmod u+x get_chatMGL.sh
./get_chatMGL.sh
```
This script requires gdown package, included in the "requirements.txt" file.

## Testing run
We provide a bash script to begin with. The script provides ChatMGL responses for the file "prompts.json". To run it:
```sh
chmod u+x run.sh
./run.sh
```
This script requires ChatMGL to be under /models/chatMGL and prompts the model with questions contained in the file "prompts.json". By default, the model is tuned to generate up to 150 new tokens. This parameter is tunable, but greatly affects the running time of the script. 

The script is a wrapper for the script gen_script_chatMGL.py, which can be run using:
```sh
cd src
python3 gen_script_chatMGL.py --model_path ../models/chatMGL/ --input_questions_path ../prompts.json --output_filename ../answers_chatMGL.json --generation_tokens 150
cd ..
```
The script sets all seeds to be 42, in order to make the provided answers reproducible. Also, the generation uses default values for topk, topp and temperature, to enhance reproducability.

Because of the way we have trained the model, it has learned to repeat the initial answer before giving the actual response, thus the generation token number should be set with this in mind.

Current generations in the repo were generate with 300 max generation tokens.

## Using ChatMGL
You can load and use ChatMGL in python to provide any prompt for any question. We list an indicative example here. 
```python
from generative_model import GenerativeModel

path = "PathToTheModel" #e.g. /models/chatMGL 
model = GenerativeModel(path)

question = "Your data science question here"
response = model.generate(question)
```

If you have a hardware accelerator, ChatMGL can perform the calculations there:
```python
from generative_model import GenerativeModel

path = "PathToTheModel" #e.g. /models/chatMGL 
devide = "YourHardwareAccelerator" #e.g. cuda
model = GenerativeModel(path, device)
```


## Datasets
We make our datasets openly available in this repo, in a JSON format.
The complete generative dataset is provided in gen_dataset_chatMGL.json and the complete reward dataset in reward_dataset_chatMGL.json. We also provide the partial test, validation and train files under /dataset/ directory.


## Other Models
We provide all other models (generative and reward) we describe in our report under /models/ directory.


## About
ChatMGL was developed as the project of the MSc. level course of EPFL: "Modern Natural Language Processing"