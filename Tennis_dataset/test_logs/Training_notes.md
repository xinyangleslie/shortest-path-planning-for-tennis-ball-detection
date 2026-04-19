

### 1 epoch Test

```bash
yolo detect train model=yolo26s.pt data="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/merged_tennis_dataset/data.yaml" epochs=1 imgsz=640 batch=4 device=0 workers=2 project="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/runs" name=debug_yolo26s
```







### YOLO 26s 30 Epoch 960

```bash
yolo detect train model=yolo26s.pt data="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/merged_tennis_dataset/data.yaml" epochs=30 imgsz=960 batch=8 device=0 workers=4 patience=10 pretrained=True project="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/runs" name=yolo26s_merged_tennis_30e
```





### YOLO 26m 30 Epoch 960

```bash
yolo detect train model=yolo26m.pt data="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/merged_tennis_dataset/data.yaml" epochs=30 imgsz=960 batch=4 device=0 workers=4 patience=10 pretrained=True project="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/runs" name=yolo26m_merged_tennis_30e
```



### YOLO 26m 60 epoch 1280

```bash
yolo detect train model=yolo26m.pt data="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/merged_tennis_dataset/data.yaml" epochs=60 imgsz=1280 batch=8 device=0 workers=4 patience=20 pretrained=True project="C:/Users/Xinyang/Desktop/AdvDeepLearning/Tennis_dataset/runs" name=yolo26m_merged_tennis
```

