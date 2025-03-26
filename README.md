## Train

https://teachablemachine.withgoogle.com/train/image/1FsMQauZdUvkO1DCY1AiGVlOUeKQYNtEN

Save as `model_edgetpu.tflite` and `labels.txt` in root directory.

## Eval

```bash
python eval.py --model model_edgetpu.tflite --dataset dataset/test/ --labels labels.txt
```

## Run Sorter

```bash
cd Sorter
python sorter.py --opencv --biquad
```
