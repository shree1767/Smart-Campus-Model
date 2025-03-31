import cv2
from ultralytics import solutions

def process_video(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    region_points = {
        "full-frame": [(0, 0), (w, 0), (w, h), (0, h)]
    }

    # Initialize region counter
    regioncounter = solutions.RegionCounter(
        show=False,
        region=region_points,
        model="yolo11n.pt",
        classes=[0]
    )

    total_count = 0

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        results = regioncounter(im0)
        
        # Extracting the count from results
        if hasattr(results, "region_counts") and "full-frame" in results.region_counts:
            total_count = max(total_count, results.region_counts["full-frame"])

    cap.release()
    return total_count