# file: demo_infer_boxes.py

import numpy as np
import torch
from PIL import Image

# Make sure you have your bounding-box SAM2ImagePredictor in the import path
from sam2.predictors.sam2_image_predictor import SAM2ImagePredictor


def main():
    """
    Demo file to illustrate bounding box inference with SAM2ImagePredictor (box version).
    """

    # 1. Instantiate the model predictor.
    #    If you have a local checkpoint or model_id on HF, specify it below.
    #    This example simply demonstrates structure, so the model_id is just a placeholder.
    model_id = "your_hf_repo_or_local_checkpoint"  
    # e.g. from_pretrained(...) might need paths or additional arguments
    predictor = SAM2ImagePredictor.from_pretrained(model_id=model_id)
    predictor.model.eval()

    # 2. Load an image (random or from file)
    #    Here we make a random image in memory for demo:
    #    Alternatively, you can load from disk: image = Image.open("my_image.jpg")
    #    Or use np array in HWC (RGB) format.
    h, w = 480, 640
    random_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # 3. Set the image in the predictor
    predictor.set_image(random_img)

    # 4. Prepare some random point prompts (just for demonstration)
    #    Suppose we generate 2 random points inside the image:
    point_coords = np.array([
        [100, 200],
        [300, 100],
    ])
    point_labels = np.array([1, 0])  # e.g. first is foreground, second is background

    # 5. Optionally, add a user box prompt
    #    For instance, [x1, y1, x2, y2] in pixel coords
    #    (Here a random rectangle, but in a real scenario you'd have an actual bounding box.)
    box_xyxy = np.array([150, 100, 350, 300])

    # 6. Predict bounding boxes
    #    Because we replaced the original mask-based code with a bounding box decoder,
    #    .predict(...) returns (boxes, scores).
    pred_boxes, pred_scores = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box_xyxy,
        multimask_output=True,   # remains for API compatibility, not used for boxes
        normalize_coords=True,   # if your pipeline expects normalized coords
    )

    # 7. Print results
    print("Predicted bounding boxes (XYXY):", pred_boxes)
    print("Object scores:", pred_scores)


if __name__ == "__main__":
    main()
