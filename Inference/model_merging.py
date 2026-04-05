from llamafactory.train.tuner import export_model

def main():
    checkpoint_path = "/scratch/hay3fm/ablation/Qwen7B/early/checkpoint-500"
    
    args = {
        "model_name_or_path": "Qwen/Qwen2-VL-7B-Instruct",
        "adapter_name_or_path": checkpoint_path,
        "template": "qwen",
        "finetuning_type": "lora",
        "export_dir": "/scratch/hay3fm/ablation/Qwen7B/merged-500",
        "export_size": 2,
        "export_device": "auto"
    }
    
    print(f"Loading base model and adapter from: {checkpoint_path}")
    print("Starting the merge process... This might take a few minutes.")
    
    export_model(args)
    
if __name__ == "__main__":
    main()