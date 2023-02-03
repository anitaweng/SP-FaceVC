# SPFaceVC

This is the offcial github page of the AAAI 2023 proceeding paper: "Zero-shot Face-based Voice Conversion: Bottleneck-Free Speech Disentanglement in the Real-world Scenario".

1. Set up environment and generate data from [WAVEGLOW](https://github.com/NVIDIA/waveglow).
   Change the mel2amp.py file to the one under SPFaceVC/ to generate numpy training and testing files.
   
2. Change the directory path to your own path in data_loader.py.
3. Train the model with command
```
python3 main_gan.py --model_id $your_id$
```
4. Generate the results with 
