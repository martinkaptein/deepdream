# Google DeepDream fork

This is my fork of [Google's Deep Dream code](https://github.com/google/deepdream) with modifications (such as updated code and comments) and extra python scripts.

Some of the scripts are still a work in progress.

> Please check comments in the code.

## Libraries and stuff

Same as [official Google DeepDream](https://github.com/google/deepdream). Double check imports and use `pip install <stuff>` if you run into problems. Python version is **Python 2.7**.

## Python scripts

More or less functional scripts are:

- script.py
- video.py
- video.convert.py
- DreamControl.py
- inception.py
- layer_key_gen.py

## Description

    script.py
    
Uses iterative process for vivid result.

    video.py
    
Creates DeepDream video with zoom effect (provided in official code).

    video-convert.py
    
Convert bunch of frames (batch) to DeepDream. Useful to convert a *normal* video to DeepDream video.

    DreamControl.py
    
Control which features are to be 'filtered'.

    inception.py
    
Rather experimental script to *dream* inside a dream.

    layer_key_gen.py
    
Print all layer names of a model.