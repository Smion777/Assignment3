import jetson.inference
import jetson.utils
import os

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.4)
image_path = "/home/nvidia/jetson-inference/data/images/object_3.jpg"

output_dir = "/home/nvidia/jetson-inference/examples"
output_filename = "detected" + os.path.basename(image_path)
output_path = os.path.join(output_dir,output_filename)

os.makedirs(output_dir,exist_ok=True)




if not os.path.exists(image_path):
    print(f"error:no images found - {image_path}")
    exit()
print("start...")
print("=" * 50)

display = jetson.utils.videoOutput("display://0")
img = jetson.utils.loadImage(image_path)
detections = net.Detect(img)
print(f"image: {os.path.basename(image_path)}")
print(f"find {len(detections)} class")

for j, detection in enumerate(detections):
    class_name = net.GetClassDesc(detection.ClassID)
    
    width = detection.Right - detection.Left
    height = detection.Bottom - detection.Top
    area = width * height
    center_x = detection.Left + width / 2
    center_y = detection.Top + height / 2
    
    print(f"\nnumber {j+1}:")
    print(f"  ClassID: {class_name} (ID: {detection.ClassID})")
    print(f"  Confidence: {detection.Confidence:.4f}")
    print(f"  Left={detection.Left:.2f}, Top={detection.Top:.2f}, Right={detection.Right:.2f}, Bottom={detection.Bottom:.2f}")
    print(f"  Width={width:.2f}, Height={height:.2f}, Area={area:.2f}")
    print(f"  Center: ({center_x:.2f}, {center_y:.2f})")


display.Render(img)
jetson.utils.saveImage(output_path, img)
print("\nSave to:{output_path}")
print("\nfinish.")

display.Close()
