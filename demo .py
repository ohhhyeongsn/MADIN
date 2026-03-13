import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import glob
from src.network import MADIN
from src.datasets import RandomMask

def main():
    parser = argparse.ArgumentParser(description="MADIN Inference Demo")
    parser.add_argument('--image', type=str, required=True, help='Path to input RGB image or folder')
    parser.add_argument('--mask', type=str, help='Path to mask image or folder. If not provided or matching file not found, a random mask is generated.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='outputs', help='Path to save the result image or output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Model
    model = MADIN().to(device)
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'net_G' in checkpoint:
        model.load_state_dict(checkpoint['net_G'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully.")

    # 2. Prepare File List
    if os.path.isdir(args.image):
        img_list = sorted(glob.glob(os.path.join(args.image, '*.*')))
        # Filter for common image extensions
        img_list = [f for f in img_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        is_dir_mode = True
        os.makedirs(args.output, exist_ok=True)
    else:
        img_list = [args.image]
        is_dir_mode = False

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    print(f"Processing {len(img_list)} images...")

    for img_path in img_list:
        try:
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            print(f"Processing: {filename}")

            # 3. Load Image
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # 4. Handle Mask
            mask_tensor = None
            mask_used_info = ""

            if args.mask:
                if os.path.isfile(args.mask):
                    # Use the same mask for all images
                    mask_pil = Image.open(args.mask).convert('L')
                    mask_tensor = transform(mask_pil).unsqueeze(0).to(device)
                    mask_used_info = "Single mask used."
                elif os.path.isdir(args.mask):
                    # Try to find matching mask by filename
                    mask_matches = glob.glob(os.path.join(args.mask, basename + ".*"))
                    if mask_matches:
                        mask_pil = Image.open(mask_matches[0]).convert('L')
                        mask_tensor = transform(mask_pil).unsqueeze(0).to(device)
                        mask_used_info = f"Matched mask: {os.path.basename(mask_matches[0])}"
                    else:
                        mask_used_info = "No matching mask found in directory, using random."
                else:
                    mask_used_info = "Mask path invalid, using random."

            if mask_tensor is None:
                mask_np = RandomMask(256)
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).to(device).float()
                if mask_used_info == "":
                    mask_used_info = "Random mask generated."

            # Ensure mask is strictly binary [0, 1]
            mask_tensor = (mask_tensor > 0.5).float()

            # 5. Inference
            with torch.no_grad():
                masked_img = img_tensor * mask_tensor
                output = model(masked_img, mask_tensor)
                output = torch.clamp(output, 0, 1)
                comp_img = output * (1.0 - mask_tensor) + img_tensor * mask_tensor

            # 6. Save Main Result
            if is_dir_mode:
                out_name = os.path.join(args.output, basename + "_out.png")
            else:
                out_name = args.output

            res_pil = transforms.ToPILImage()(comp_img.squeeze(0).cpu())
            res_pil.save(out_name)

            print(f"  [{mask_used_info}] -> Saved to {out_name}")

        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

    print("\nAll tasks finished successfully!")

if __name__ == '__main__':
    main()
