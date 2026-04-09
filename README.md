<img src="./Images/VLM_Fig1.pdf" width="500px"></img>

## Synoptic-Bench

Recent advances in visual-language models (VLMs) have led to significant improvements in a plethora of complex multimodal tasks like image captioning, report generation, and visual perception. However, generating text from meteorological data is highly challenging because the atmosphere is a chaotic system that is rapidly changing at various spatial and temporal scales. Given the complexity of atmospheric phenomena, it is critical to verifiably quantify the effectiveness of existing VLMs on weather forecasting data. In this work, we present Synoptic-Bench, a high-quality dataset consisting of 1,367,041 text samples of Advanced Forecast Discussions created by the National Weather Service over the continental United States paired to images of 500mb geopotential height, 2 meter temperature, and 850mb wind velocity in weather forecasts. We also present Synoptic Phenomena Alignment and Coverage Evaluation (SPACE), a novel evaluation framework that can be used to effectively estimate the quality of text descriptions of synoptic weather phenomena. Extensive experiments on generating forecast discussions using state-of-the-art VLMs show the sensitivity of existing evaluation metrics in this domain and enable further exploration into synoptic weather and climate text generation.

## Contents
- [Data and Weights](#Pretrained_models)
- [Train](#train)
- [SPACE Evaluation](#SPACE)
- [Traditional Evaluation](#Traditional_Eval)
- [Preprocessing Setup](#preprocessing)



## Data and Model Weights

The full dataset including saved model weights can be found at https://huggingface.co/datasets/Aikyam-Lab/Synoptic-Bench.

## Train

We finetune LLaVA-v1.5-7B, LLaVA-v1.5-13B, Qwen3-VL-7B, and LLaMA-3.2-11B with 1 NVIDIA H200 GPU. The code and training parameters to train each model is in the "train" folder.

## Evaluation

We use Synoptic Phenomena Alignment and Coverage Evaluation (SPACE) for evaluation. The code to run SPACE is in the "SPACE" folder.

## Preprocessing Setup

Step 1: 

