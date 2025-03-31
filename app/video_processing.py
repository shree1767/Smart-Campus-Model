import cv2
from ultralytics import solutions

def process_video(video_path: str) -> tuple:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4

    # Define output video path
    output_path = video_path.replace(".mp4", "_processed.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

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

        # Extract count
        if hasattr(results, "region_counts") and "full-frame" in results.region_counts:
            total_count = max(total_count, results.region_counts["full-frame"])

        # Draw detections on frame
        if hasattr(results, "plot"):
            im0 = results.plot()

        # Write frame to output video
        out.write(im0)

    cap.release()
    out.release()

    return total_count, output_path