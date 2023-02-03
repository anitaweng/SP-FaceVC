# SPFaceVC

This is the offcial github page of the AAAI 2023 proceeding paper: "**Zero-shot Face-based Voice Conversion: Bottleneck-Free Speech Disentanglement in the Real-world Scenario**".

The demo website: [https://sites.google.com/view/spfacevc-demo](https://sites.google.com/view/spfacevc-demo)

1. Change input and output path in ***preprocess/config/preprocess.yaml*** and  generate data with command:
```
python3 preprocess/preprocess.py
```
2. Change the directory path to your own path in ***data_loader.py***.
3. Train the model with command:
```
python3 main_gan.py --model_id $your_id$
```
4. Change the parameters in the file to your checkpoint and data, then generate results with command:
```
python3 conversion_speechbrain.py
```
5. Synthesized results with [WAVEGLOW](https://github.com/NVIDIA/waveglow) pretrained model and change the ***inference.py*** to our file, running with command:
```
python3 inference.py -f $your_result_path$ -w $waveglow_checkpoint_path$ -o $output_dir$ --is_fp16 -s 0.6
```
