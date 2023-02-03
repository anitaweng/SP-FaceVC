# SPFaceVC

This is the offcial github page of the AAAI 2023 proceeding paper: "**Zero-shot Face-based Voice Conversion: Bottleneck-Free Speech Disentanglement in the Real-world Scenario**".

The demo website: [https://sites.google.com/view/spfacevc-demo](https://sites.google.com/view/spfacevc-demo)

0. Download your data (with face and speech).
1. Change wav input and output path in ```preprocess/config/preprocess.yaml``` and generate data with command:
```
python3 preprocess/preprocess.py
```
2. Change ```rootDir``` and ```targetDir``` in ```make_faceemb.py```, and execute to get face embedding with command:
```
python3 make_faceemb.py
```
Then, doing arithmatic mean for the embeddings. (Change your input and output dir path as well)
```
python3 make_spk_mean.py
```
3. Change the directory path to your own path in ```data_loader.py```.
4. Train the model with command:
```
python3 main_gan.py --model_id $your_id$
```
5. Change the parameters in the file to your checkpoint and data, then generate results with command:
```
python3 conversion_speechbrain.py
```
6. Synthesized results with [WAVEGLOW](https://github.com/NVIDIA/waveglow) pretrained model and change the ***inference.py*** to our file, running with command:
```
python3 inference.py -f $your_result_path$ -w $waveglow_checkpoint_path$ -o $output_dir$ --is_fp16 -s 0.6
```
