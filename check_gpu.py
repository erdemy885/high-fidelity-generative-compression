
import torch

def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available!")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU is not available. PyTorch will use CPU.")

if __name__ == "__main__":
    check_gpu()
