from ultralytics import YOLO
import torch


def main():
    # 自動判斷是否可用 GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 載入訓練完成後的最佳模型
    model = YOLO(
        r"C:\Users\LSS\Documents\114下\AI\AI-Hw1\Yolo26\runs\parking_occupancy_yolo262\weights\best.pt"
    )

    # 執行驗證
    metrics = model.val(
        # 指定 data.yaml
        data=r"C:\Users\LSS\Documents\114下\AI\AI-Hw1\Yolo26\dataset\data.yaml",

        # 指定裝置
        device=device,

        # 驗證影像大小
        imgsz=640,

        # 指定要驗證的資料切分
        # 可選 "val" 或 "test"
        split="val"
    )

    # 印出驗證結果
    print("Validation finished.")
    print(metrics)


if __name__ == "__main__":
    main()