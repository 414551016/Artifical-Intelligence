# BERT
BERT（Bidirectional Encoder Representations from Transformers）是由 Google 在 2018 年提出的一種自然語言處理（NLP）預訓練模型，它徹底改變了語言理解任務的表現。
- Bidirectional（雙向）
  同時考慮「前後文」
- Encoder
  使用 Transformer 的 encoder 結構（不是 decoder）
- Representations
  學習詞語的語意表示（embedding）
- Transformers
  基於 Transformer 架構（Attention 機制）

## BERT 是什麼？
BERT = 一種基於 Transformer 的雙向語言表示模型，它的核心目標是：讓電腦更「理解句子的上下文語意」

## 核心特色（最重要）
### 雙向理解（Bidirectional Context）
- 傳統模型（如 LSTM、單向語言模型）：只能從左→右 或 右→左讀句子
- BERT：同時看「前後文」
  例子："He sat by the bank"。bank 可能是「河岸」或「銀行」，BERT 會用前後文判斷正確意思。
### Transformer 架構
- BERT 使用：
  - Self-Attention 機制
  - Encoder 堆疊
- 優點：
  - 可並行處理（比 RNN 快）
  - 能捕捉長距離依賴
### 預訓練 + 微調（Pretraining + Fine-tuning）
- Step 1：預訓練（Pretraining）
  用大量未標註文本學習語言
  - 👉 任務（就是 pretext tasks）：
    - Masked Language Model (MLM)
    - Next Sentence Prediction (NSP)
- Step 2：微調（Fine-tuning）
  在特定任務上做小量訓練，例如：
  - 文本分類
  - 問答系統
  - 情感分析
  - 命名實體辨識（NER）

## BERT 的兩大訓練任務
- Masked Language Model（MLM）
  隨機遮住句子中的某些詞，讓模型預測。例子：I love [MASK] learning. 模型要預測：👉 "machine" 或 "deep"
  - 功能：
    - 學語意關係
    - 理解上下文
- Next Sentence Prediction（NSP）
   判斷兩句話是否連續。例：句子 A + 句子 B → 判斷 B 是否接在 A 後面
  - 功能：
    - 理解句子間關係
