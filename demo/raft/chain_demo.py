import re
import argparse
from pathlib import Path
import cv2
import numpy as np
from sakuramoti.flow_model.raft import RAFT
from sakuramoti.visualizer.flow_vis import flow_to_image
from sakuramoti.io import load_image
from sakuramoti.transformation import InputPadder

def viz(img, flo):
    img = 0.5 *  255.0 * (1 + img)
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    cv2.imshow("image", img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()

def chain_demo(path:str, device:str):
    
    model = RAFT().to(device=device).eval()
    pattern = re.compile(r'.*\.(png|jpg)$', re.IGNORECASE)
    images = [str(file) for file in Path(path).iterdir() if pattern.match(str(file))]
    images = sorted(images)
    
    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = load_image(imfile1, device)
        image2 = load_image(imfile2, device)
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
            
        _, flow_up = model.pred(image1, image2, iters=20)      
        viz(image1, flow_up)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="dataset for evaluation")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], help="use cuda device")
    args = parser.parse_args()
    chain_demo(args.path, args.device)