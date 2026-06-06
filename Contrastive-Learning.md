# Contrastive Learning (對比學習)
對比學習（Contrastive Learning），屬於自監督學習的一種方法，主要用於學習資料的表示（representation）。它的核心思想很直觀：**讓相似的資料在向量空間中更接近，不相似的資料更遠離**。一句話總結：對比學習就是「透過比較來學習」，讓模型知道誰應該像誰、誰不應該像誰。

## 基本概念
對比學習會把資料轉換成向量（embedding），然後透過「比較」不同樣本之間的關係來學習。
- 在訓練過程中通常會用三種資料關係：
  給定一組相似（Similar，或稱正樣本）與相異（Dissimilar，或稱負樣本）的資料樣本對 。
  - 正樣本（Similar, Positive pair）
    兩個本質上相同或相關的資料。例如：**同一張圖片**的**不同裁切版本**
  - 負樣本（Dissimilar, Negative pair）
    不相關的資料。例如：**不同圖片**
- 模型目標：
  讓相似的資料在向量空間中更接近，不相似的資料更遠離。
  - 拉近正樣本距離。
  - 拉遠負樣本距離。

## 常見方法：
- Siamese Network（雙塔網路）
- Triplet Loss（最經典）
- InfoNCE / SimCLR（現代主流）

## 優點 / 缺點
- 優點：
  - 不需要標註資料（Self-supervised）
  - 可以學到通用特徵（representation）
  - 適用多種領域：
    - 電腦視覺（image embedding）
      - 圖像搜尋（找相似圖片）
      - 人臉辨識
    - NLP（句子 embedding）
      - Sentence embedding（像 BERT embedding）
      - 相似句子搜尋
    - 推薦系統
      - 找相似商品
      - 用戶興趣建模
    - 語音


