import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import Model


class GroundingSAM2:
    """
    A class that combines Grounding DINO and SAM2 for instance segmentation with text prompts.
    """
    
    def __init__(
        self,
        sam2_model_config: str,
        sam2_checkpoint: str,
        grounding_dino_config: str,
        grounding_dino_checkpoint: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = "outputs/grounding_sam2",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        use_amp: bool = False,  # Added parameter to control automatic mixed precision
    ):
        """
        Initialize the GroundingSAM2 model with configurations.
        
        Args:
            sam2_model_config: Path to SAM2 model configuration
            sam2_checkpoint: Path to SAM2 model checkpoint
            grounding_dino_config: Path to Grounding DINO config
            grounding_dino_checkpoint: Path to Grounding DINO checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
            output_dir: Directory to save outputs
            box_threshold: Box detection confidence threshold
            text_threshold: Text detection confidence threshold
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # Initialize SAM2
        self.sam2_model = build_sam2(sam2_model_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        # Initialize Grounding DINO
        self.grounding_model = Model(
            model_config_path=grounding_dino_config,
            model_checkpoint_path=grounding_dino_checkpoint,
            device=device
        )
        
        # Set up mixed precision if available
        if device == "cuda" and use_amp:
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
    
    def process_image(
        self, 
        image_path: str = None, 
        image: np.ndarray = None, 
        classes: List[str] = None,
        output_dir: str = None,
        return_results: bool = True
    ) -> Dict:
        """
        Process an image with grounding and segmentation.
        
        Args:
            image_path: Path to the input image file (optional if image is provided)
            image: Numpy array of the input image (optional if image_path is provided)
            classes: List of class names to detect
            output_dir: Custom output directory for this specific run
            return_results: Whether to return the results as a dictionary
            
        Returns:
            Dict containing detection and segmentation results
        """
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided")
            
        if image is None:
            # Load the image
            image = cv2.imread(image_path)
        else:
            image_path = "provided_image"
            
        if classes is None:
            raise ValueError("Classes list must be provided")
            
        # Set output directory for this run
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.output_dir
            
        # Prepare the image for SAM2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam2_predictor.set_image(image_rgb)
        
        # Run Grounding DINO detection
        detections = self.grounding_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        
        # Extract boxes for SAM2
        input_boxes = detections.xyxy
        
        # Run SAM2 segmentation
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        
        # Post-process masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        # Get class names based on class IDs
        class_names = [classes[class_id] if class_id is not None else "unknown" 
                      for class_id in detections.class_id]
        confidences = detections.confidence.tolist()
        
        # Create formatted labels
        labels = [f"{class_name} {confidence:.2f}" 
                 for class_name, confidence in zip(class_names, confidences)]
        
        # Create detections with masks
        h, w, _ = image.shape
        detections_with_masks = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=detections.class_id
        )
        
        # Store results (optional)
        if return_results:
            # Convert masks to RLE format
            mask_rles = [self.single_mask_to_rle(mask) for mask in masks]
            
            results = {
                "image_path": image_path,
                "annotations": [
                    {
                        "class_name": class_name,
                        "bbox": box.tolist() if isinstance(box, np.ndarray) else box,
                        "segmentation": mask_rle,
                        "score": float(score),
                    }
                    for class_name, box, mask_rle, score 
                    in zip(class_names, input_boxes, mask_rles, scores)
                ],
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
                "detections": detections_with_masks,
                "labels": labels
            }
            
            return results
        
        return None

    def visualize_results(
        self, 
        results: Dict, 
        output_dir: str = None,
        filename_prefix: str = "grounded_sam2"
    ) -> np.ndarray:
        """
        Visualize detection and segmentation results.
        
        Args:
            results: Results dictionary from process_image
            output_dir: Directory to save visualization
            filename_prefix: Prefix for output filenames
            
        Returns:
            Annotated image with boxes, labels and masks
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        image_path = results["image_path"]
        detections = results["detections"]
        labels = results["labels"]
        
        # Load the image
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
        else:
            # If we are working with a provided image array
            # Get the first image from the results somehow
            # This would need to be adapted based on how you store the image
            raise ValueError("Cannot visualize results without access to the original image")
            
        # Draw bounding boxes
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
        
        # Add labels
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        
        # Add masks
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame, detections=detections)
        
        # Save annotated image
        box_path = os.path.join(output_dir, f"{filename_prefix}_boxes.jpg")
        full_path = os.path.join(output_dir, f"{filename_prefix}_full.jpg")
        
        # Save the annotated frame with both bounding boxes and masks
        cv2.imwrite(full_path, annotated_frame)
        
        return annotated_frame
    
    def save_results(
        self, 
        results: Dict, 
        output_path: str = None
    ) -> None:
        """
        Save detection and segmentation results to a JSON file.
        
        Args:
            results: Results dictionary from process_image
            output_path: Path to save the JSON file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "grounded_sam2_results.json")
            
        # Extract data to save (remove detections object as it's not JSON-serializable)
        serializable_results = {
            "image_path": results["image_path"],
            "annotations": results["annotations"],
            "box_format": results["box_format"],
            "img_width": results["img_width"],
            "img_height": results["img_height"]
        }
        
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=4)
    
    @staticmethod
    def single_mask_to_rle(mask):
        """Convert a binary mask to RLE format."""
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle


# Example usage
if __name__ == "__main__":
    # Configuration
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    GROUNDING_DINO_CHECKPOINT = "/home/yutian/temp_projects/Grounding-Dino-FineTuning/weights/model_weights200.pth"
    
    # Initialize the model
    model = GroundingSAM2(
        sam2_model_config=SAM2_MODEL_CONFIG,
        sam2_checkpoint=SAM2_CHECKPOINT,
        grounding_dino_config=GROUNDING_DINO_CONFIG,
        grounding_dino_checkpoint=GROUNDING_DINO_CHECKPOINT,
        output_dir="outputs/example_run"
    )
    
    # Process an image
    results = model.process_image(
        image_path="pepper.jpg",
        classes=["peduncle", "fruit"]
    )
    
    # Visualize results
    annotated_image = model.visualize_results(results)
    
    # Save results to JSON
    model.save_results(results)
    print(f"results: {results['labels']}")