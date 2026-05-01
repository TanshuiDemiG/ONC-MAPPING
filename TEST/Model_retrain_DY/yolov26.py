from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")
    results = model.train(data="data.yaml", epochs=1000, imgsz=512,device=0)

if __name__ == '__main__':
    main()