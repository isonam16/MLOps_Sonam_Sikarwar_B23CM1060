# from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_face_drag, create_demo_paste
import cv2
import numpy as np
import torch
# from src.demo.download import download_all
from DragonDiffusion.src.demo.model import DragonModels
# from src.utils.utils import resize_numpy_image
from torchvision.transforms import PILToTensor
from PIL import Image
import time

# download_all()

import cv2

# main demo
# pretrained_model_path = "runwayml/stable-diffusion-v1-5"
# model = DragonModels(pretrained_model_path=pretrained_model_path)


def move(img,mask,x_s,y_s,x_e,y_e, model):#@cherry pass mask as numpy 

    #make sure img is numpy read from cv2 not from PIL
    #@harsh  pass any coordinates user select in the object as x_s and y_s and the position to move x_e and y_e
    
    dy=y_e-y_s
    dx=x_e-x_s
    prompt="" #if you do object detetion then give with label else no need 
    return model.run_move(img,mask,dx,dy)[0]


if __name__=="__main__":
    img=cv2.imread("/home/raid/adobe/MLpipeline/data/original_from_frontend/image_1764336305970.png")
    

    mask=cv2.imread("/home/raid/adobe/MLpipeline/data/masks_for_frontend/combined_1764336306025_mask_0.png")
    s_t=time.time()
    img_out=move(img,mask,488,418,653,408)
    print(time.time()-s_t)
    
    cv2.imwrite("res.png",img_out)