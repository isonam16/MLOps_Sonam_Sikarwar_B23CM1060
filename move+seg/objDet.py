import cv2
import os
import numpy as np
import pandas as pd


def run_object_detection(img_path , x, y , model=None, output_image_dir="/app/data/bb"):
    """
    Runs YOLO object detection and returns bounding boxes as a list of dicts.
    """

    results = model(img_path)
    os.makedirs(output_image_dir, exist_ok=True)
    img = cv2.imread(img_path)

    detections = [] 
    csv_path="/app/data/detections_log.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=[
            "image", "class_id", "confidence", "left", "top", "right", "bottom"
        ])

    label = None

    for r in results:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()       # shape (N, 4)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):
            left, top, right, bottom = xyxy[i]
            class_id = cls[i]
            confidence = float(conf[i])
            
            detections.append({
                "image": os.path.basename(r.path),
                "class_id": class_id,
                "confidence": confidence,
                "left": float(left),
                "top": float(top),
                "right": float(right),
                "bottom": float(bottom)
            })

            color = (0, 255, 0)
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), color, 2)
            label = class_id
                # cv2.putText(img, label, (int(left), int(top) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            if x>left and x<right and y>top and y<bottom:
                print(f"Coordinate ({x},{y}) is inside bounding box ID:{class_id} with confidence {confidence:.2f}")
                label = class_id

    # Save annotated image
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_image_dir, f"{filename}_boxes.jpg")
    cv2.imwrite(output_path, img)
    print(f"Saved image with boxes at {output_path}")

    df_new = pd.DataFrame(detections)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.drop_duplicates(
        subset=["image", "class_id", "left", "top", "right", "bottom"],
        keep="first",
        inplace=True
    )
    df_combined.to_csv(csv_path, index=False)
    return detections, output_path, label

# def                                                                                                                                 generate_masks(left: float, top: float, right: float, bottom: float,
#                    img_height: int, img_width: int,
#                    original_image_path: str,
#                    output_dir: str) -> str:
 
#     mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
#     cv2.rectangle(mask,
#                   (int(left), int(top)),
#                   (int(right), int(bottom)),
#                   color=255,
#                   thickness=-1)

#     mask=255-mask  # Invert mask if needed
    
#     # derive filename for the mask
#     base_name = os.path.basename(original_image_path)           
#     name, ext = os.path.splitext(base_name)                    
#     mask_filename = f"{name}_mask{ext}"                         
    
#     # full path
#     out_path = os.path.join(output_dir, mask_filename)
    
#     # save the mask
#     success = cv2.imwrite(out_path, mask)
#     if not success:
#         raise IOError(f"Could not write mask image to {out_path}")
    
#     return out_path
