# How haptic / force sensory is used to reduce harm
```
時間：4 分鐘
Presenter：劉順松

這一段說明 觸覺或力感測如何降低傷害。

Haptic / force sensory 指的是機器人透過力感測器、扭矩感測器、觸覺感測器或接觸回饋，知道自己是否碰到人或物體，以及施加的力量是否過大。

可說明的內容包括：
  機器人偵測接觸力；
  當碰撞或過大力量發生時立即停止；
  控制機器人以柔順方式移動；
  在共享工作空間中降低人類受傷風險；
  補足 vision 的盲點，例如攝影機看不到或被遮擋時。

這段的核心訊息是：
vision 可以提前看見危險，但 haptic / force sensing 可以在接觸發生時降低實際傷害。

我的看法：
這段放得很好，因為它讓報告不只停留在 vision 和 language，而是納入實體機器人安全中非常重要的 sensing modality。建議你可以強調 force sensing 是「最後一道安全防線」：即使視覺判斷失敗，力感測仍能在接觸時保護人類。
```

## 本次任務報告內容彙整
你的段落時間是 4 分鐘，建議做 4 張投影片，每張約 1 分鐘。
- 投影片 1：為什麼觸覺/力覺感知如此重要
```
Slide 1：Why haptic / force sensing matters(投影片 1：為什麼觸覺/力覺感知如此重要)
投影片標題
  Haptic / Force Sensing: The Last Safety Layer in HRC(觸覺/力感應：人機互動的最後一道安全防線)
投影片重點
  在人機協作中，vision 可以提前看到危險，verbal communication 可以讓人類提醒機器人修正錯誤，但真正發生接觸時，機器人仍然需要 haptic / force sensing 來降低傷害。
可以放這張表：
技術	                發生時間	    功能
Vision	              接觸前	      預測人類位置、避免碰撞
Verbal communication	錯誤發生時	  人類提醒機器人修正
Haptic/force sensing	接觸發生時	  偵測碰撞、限制力量、降低傷害
講稿建議
My part focuses on haptic and force sensing. In HRC, vision can help prevent accidents before contact, and verbal communication can help correct robot mistakes. However, when physical contact actually happens, the robot needs force or tactile sensing to detect the contact and react immediately. Therefore, haptic sensing can be considered the last safety layer in human-robot collaboration.
我的部分將重點放在觸覺和力覺感知。在人機協作中，視覺可以幫助預防接觸前的意外，而語音交流可以幫助糾正機器人的錯誤。然而，當實際發生物理接觸時，機器人需要力覺或觸覺感知來偵測接觸並立即做出反應。因此，觸覺感知可以被視為人機協作的最後一道安全防線。
```
- 投影片 2：什麼是觸覺和力覺感測器？
```
Slide 2：What are haptic and force sensors?
投影片標題
  What Does the Robot Sense?(機器人感知什麼？)
投影片重點
  Haptic / force sensing 可以分成幾種類型：
感測方式	                  感測內容	          用途
  Force / torque sensor	    末端受力與扭矩	    偵測抓取、碰撞、推拉
  Joint torque sensing	    關節外力矩	        判斷哪個關節受到外力
  Tactile sensor/robot skin	接觸位置與壓力分布	判斷碰到哪裡、壓力多大
  Motor current estimation	用馬達電流估測外力	低成本碰撞偵測
你可以補一句：
  機器人不一定要有真正的「皮膚」，也可以透過關節力矩或馬達電流估測外部接觸。
講稿建議
Haptic sensing does not only mean artificial skin. It includes force-torque sensors at the wrist, joint torque sensors, tactile sensors on the robot surface, and even external torque estimated from motor currents. These signals tell the robot whether it is touching a person, where the contact may occur, and how strong the contact force is.
觸覺感知不僅僅指人造皮膚。它包括手腕處的力矩感測器、關節扭矩感測器、機器人表面的觸覺感測器，甚至包括根據馬達電流估算的外部扭矩。這些訊號告訴機器人它是否正在接觸人、接觸可能發生的位置以及接觸力的大小。
```
- 投影片 3：力覺感知如何減少傷害
```
Slide 3：How force sensing reduces harm
投影片標題
  From Contact Detection to Harm Reduction(從接觸檢測到傷害減少)
投影片重點
  這張是核心。建議畫成流程：
  Contact occurs → Detect collision → Estimate force/location → React safely
接觸發生 → 偵測碰撞 → 估計力/位置 → 安全應對
可以整理為四步：
  步驟	          問題	        機器人反應
  Detection	      是否碰到人？	  停止或減速
  Isolation	      碰撞在哪裡？	  避開接觸方向
  Identification	力有多大？	    限制力與壓力
  Reaction	      如何降低傷害？	退讓、降低剛性、重新規劃
這裡可以引用 T-RO 2017 的 collision survey。該論文將 collision 問題分成 detection、isolation、identification，並討論使用機器人自身感測資訊進行即時碰撞分析。
講稿建議
The key idea is not only to stop the robot after collision. A complete safety pipeline includes collision detection, collision isolation, force identification, and safe reaction. First, the robot detects abnormal external force. Then, it estimates where the contact happens and how large the force is. Finally, it reacts by stopping, moving away, reducing stiffness, or replanning the trajectory.
關鍵在於碰撞後機器人不僅停止運動。完整的安全流程包括碰撞偵測、碰撞隔離、力識別和安全回應。首先，機器人檢測異常的外力。然後，它估計接觸發生的位置和力的大小。最後，它透過停止、遠離、降低剛度或重新規劃軌跡來做出反應。
```
- 投影片 4：回應策略及其與信任的關係
```
Slide 4：Reaction strategies and relation to trust
投影片標題
  Safe Reaction Builds Trust(安全回應建立信任)
投影片重點
  Force sensing 之後，機器人可以採取以下反應：
  反應策略	            說明	        對人類感受
  Emergency stop	    立即停止	      安全，但可能中斷任務
  Speed reduction	    降低速度	      減少衝擊能量
  Force limiting	    限制接觸力	    避免超過人體容許值
  Compliance control	變柔順	        不會硬推人類
  Retreat / replan	  退讓或重新規劃	顯得更可控、更可信任

這裡可以把你的段落接回全組主題：
haptic / force sensing 不只是降低物理傷害，也提升 psychological safety。
本組已上傳的 HRI/C survey 也提到，force sensors 在 HRC 中常與 vision 一起用於 object handover、collaborative manipulation，以及安全工作空間中的 contact detection；tactile sensors 也可用於 object handover，輔助與環境的實體互動。
另一篇你們組內候選論文也指出，現代 cobots 會結合 force and torque sensors、force limits 與 anti-collision systems，以提升安全有效的人機協作。
講稿建議
These reactions are important not only for physical safety, but also for trust. If the robot keeps pushing after touching a human, the user will immediately lose trust. But if the robot detects contact, reduces force, and moves away smoothly, the user can feel that the robot is aware, controllable, and safe to work with.
這些回應不僅對人身安全至關重要，對建立信任也同樣重要。如果機器人在接觸人後繼續推擠，用戶會立即失去信任。但如果機器人偵測到接觸後，降低力並且平穩地移動，使用者就會感受到機器人具有感知能力、可控制性，並且安全可靠。
```
