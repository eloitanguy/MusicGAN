# Touhou music generation using GANs

## Presentation

Using a few Touhou music MIDIs, this project trains simple Generative Adversarial Networks in order to generate music.

## Setup

In order to install the required python packages, use:

    pip install requirements.txt

In order to prepare the music dataset, use the command:
    
    cd data
	python midi.py

## Training

All the parameters can be changed in the `config.py` file. Afterwards simply run the following command (from the main repository):

    python train.py

This will create a checkpoint folder in checkpoints/[experiment name]/. Warning: the code was written for GPU training, if you don't have a GPU or too little VRAM, you will have to remove all the `.cuda()` from the code.

If you want to listen to created music (MIDI format) from the training that just finished, run:

    python test.py

If you want to generate music using a previous training, you can specify the .pth checkpoint path:

    python test.py --checkpoint [your G.pth checkpoint]
