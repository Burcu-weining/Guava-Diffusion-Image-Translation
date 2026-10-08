# Guava Image Translation with a Conditional Diffusion Experiment

A PyTorch image-to-image experiment transforming photos of **wrapped guavas** into approximate **unwrapped guava** images. The training and validation photographs were collected by the project author.

> Experimental project: the dataset is very small and the generation/evaluation results are preliminary. The architecture is a compact convolutional noise predictor, rather than a full multiscale U-Net.

## Dataset

The repository contains **40 JPEG photographs arranged into 20 matched pairs**:

| Split | Wrapped images | Unwrapped images | Pairs |
| --- | ---: | ---: | ---: |
| Train | 10 | 10 | 10 |
| Validation | 10 | 10 | 10 |

Files are provided under `dataset/train/{wrapped,unwrapped}` and `dataset/val/{wrapped,unwrapped}`. Matching filenames define the paired examples.

## Architecture and training

- PyTorch conditional image-generation experiment using a convolutional encoder/decoder noise predictor
- Input: noisy target concatenated with wrapped conditioning image
- Linear beta schedule and iterative reverse sampling
- Image size **128 × 128**; batch size **4**; **50** diffusion timesteps; **8** configured training epochs
- Adam optimizer with learning rate **0.0001**
- Combined MSE noise-prediction and L1 reconstruction proxy loss

## Original notebook results

These values were **recorded in the original notebook's saved evaluation output**, and have **not been independently reproduced** for this packaged version:

| Metric (validation mean) | Original recorded value |
| --- | ---: |
| SSIM | **0.04077663** |
| PSNR | **7.75102734 dB** |
| L1 error | **0.38922936** |

The `Diffusion.ipynb` notebook visibly records these same historical metrics in a markdown cell and retains its original saved training logs and image-comparison output. After training, its evaluation cell **prints newly computed SSIM, PSNR and L1 values**. Results may vary across runs. The original metrics indicate weak reconstruction quality on this small experiment.

## Run locally

```bash
git https://github.com/Burcu-weining/Guava-Diffusion-Image-Translation.git
cd Guava-Diffusion-Image-Translation
python -m pip install -r requirements.txt
```

Open `Diffusion.ipynb` in VS Code, Jupyter, or Colab. Run the cells in order to load the local `dataset/` directory, train the model, generate comparisons, and print the evaluation metrics. Training and evaluation can be slow on CPU. The notebook uses CUDA where available, then Apple MPS if available, otherwise CPU.

**Note:** Model weights from the original experiment were not included in the submitted notebook, so regenerating its exact original outputs may not be possible without retraining. The saved results are included as historical evidence, not as a claim of a verified reproduction.

## Limitations

Only 10 training pairs and 10 validation pairs were used. This is insufficient to establish broad generalization. The current reverse sampler and lightweight architecture are simplified relative to standard diffusion models. The notebook retains the original implementation for transparency.
