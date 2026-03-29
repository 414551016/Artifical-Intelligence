from ultralytics import YOLO
import torch


def main():
    # 自動判斷是否可使用 GPU
    # 如果 torch.cuda.is_available() 為 True，則使用第 0 張 GPU
    # 否則改用 CPU
    device = 0 if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 載入 YOLO26 預訓練模型
    # 可依需求改成 yolo26s.pt、yolo26m.pt 等
    # GTX 1650 建議先用 yolo26n.pt
    model = YOLO("yolo26n.pt")

    # 開始訓練
    results = model.train(
        # data.yaml 路徑
        data=r"C:\Users\LSS\Documents\114下\AI\AI-Hw1\Yolo26\dataset\data.yaml",

        # 訓練 epoch 數
        epochs=100,

        # 輸入影像大小
        imgsz=640,

        # batch size
        # GTX 1650 4GB 建議先用 4，若顯存不足可改 2
        batch=4,

        # 訓練裝置
        device=device,

        # DataLoader worker 數
        # Windows 建議先保守設 2，若不穩可設 0
        workers=2,

        # 訓練結果輸出資料夾
        project=r"C:\Users\LSS\Documents\114下\AI\AI-Hw1\Yolo26\runs",

        # 本次實驗名稱
        name="parking_occupancy_yolo26",

        # 是否使用預訓練權重
        pretrained=True,

        # Early stopping 的 patience
        # 若連續 20 個 epoch 沒有改善，就停止訓練
        patience=20,

        # 儲存模型
        save=True,

        # 繪製訓練曲線與結果圖
        plots=True
    )

    # 顯示訓練完成訊息
    print("Training finished.")

    # 印出訓練回傳結果物件
    print(results)

    # 這裡不再額外呼叫 model.val()
    # 原因：
    # 訓練過程中 YOLO 已經自動做 validation
    # 你前面遇到的 Invalid device id 就是出現在 train 後再手動 val()
    # 所以建議把驗證獨立成另一支程式 yolo26_val.py


if __name__ == "__main__":
    main()