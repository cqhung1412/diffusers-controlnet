# diffusers-controlnet

## Reference

In the first line of each .py file, you can find the HuggingFace URL from where the code is. 

## Requirements

- A working PC with a Nvidia GPU that has CUDA cores
- Installed CUDA environment
- Installed Python environment

## Setup

```bash
python -m venv ./env
source ./env/bin/activate
pip install -r requirements.txt
```

## How to run

- Choose a ControlNet model, i.e. canny edge in canny.py
- Update your input/output image paths
- Add your prompt, negative prompt, resolution if necessary
- Run `python canny.py`
- For first run, required checkpoints will be installed for a while
- Output will be saved in your set path