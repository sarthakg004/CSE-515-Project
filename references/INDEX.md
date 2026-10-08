# Reference Index

19 PDFs are stored in this folder. Everything here is open access, so there is nothing to pay for and nothing behind a paywall. The items marked "not a PDF" are links only, because standards and product pages cannot be redistributed.

Slide 10 asks for technical papers, industry standards, and products. The groups below map onto exactly that.

## Group A: Foundation and background

These give the vocabulary and the history. Read these first.

| Paper | Year | Why it matters |
|---|---|---|
| `Survey_Commodity_WiFi_Sensing_10_Years_2021` | 2021 | The best single overview of the whole field. Start here. |
| `SenseFi_Benchmark_WiFi_Human_Sensing_2022` | 2022 | The benchmark and code we are building on. Defines the datasets and the eleven models. |
| `Toward_ISAC_in_IEEE_802.11bf_WiFi_Networks_2022` | 2022 | Explains the 802.11bf standard and how sensing is folded into the WiFi MAC layer. |

## Group B: The core problem, making models generalise

A model trained in one room usually fails in another room. This is the open problem our end term phase attacks.

| Paper | Year | Why it matters |
|---|---|---|
| `Data_Augmentation_CrossDomain_WiFi_CSI_HAR_2024` | 2024 | Concrete augmentation recipes for the cross domain problem. Directly reusable. |
| `Self_Supervised_Learning_WiFi_CSI_HAR_Systematic_Study_2023` | 2023 | Systematic comparison of self supervised methods, so we do not have to guess. |
| `Context_Aware_Predictive_Coding_WiFi_Sensing_2024` | 2024 | A representation learning approach that reduces the need for labels. |
| `MU_SHOT_Fi_SelfSupervised_MultiUser_WiFi_Sensing_2026` | 2026 | Very recent domain adaptation without needing the source data. |
| `Identity_Agnostic_MultiUser_WiFi_Sensing_2026` | 2026 | Handles more than one person in the room, which most older work ignores. |

## Group C: Making models small and cheap enough to deploy

Sensing that costs too much computation never ships. This is the second half of our contribution.

| Paper | Year | Why it matters |
|---|---|---|
| `EfficientFi_Lightweight_WiFi_Sensing_CSI_Compression_2022` | 2022 | The standard reference for compressing CSI before it leaves the device. |
| `WiFlexFormer_Efficient_WiFi_Person_Centric_Sensing_2024` | 2024 | A small transformer designed for this exact task. |
| `Physics_Guided_Attention_Lightweight_TCN_WiFi_HAR_2026` | 2026 | Uses signal physics to shrink the model instead of brute force. |
| `WiSpike_Low_Power_Spiking_Neural_Network_WiFi_HAR_2026` | 2026 | Spiking network for very low power sensing. Good novelty angle. |
| `Resource_Efficient_WiFi_CSI_Sensing_Age_of_Samples_2026` | 2026 | Treats how often you sample as a resource to be budgeted. Strong networking angle. |
| `maxVSTAR_Vision_Guided_CSI_Edge_Adaptation_2025` | 2025 | Adapts the model on the edge device itself after deployment. |

## Group D: Baselines, localization, and adjacent work

| Paper | Year | Why it matters |
|---|---|---|
| `Evaluating_BiLSTM_CNN_GRU_WiFi_CSI_HAR_2025` | 2025 | A clean recent baseline comparison. Useful sanity check for our own numbers. |
| `Lessons_Deploying_CSI_Localization_LargeScale_ISAC_2025` | 2025 | What actually breaks when CSI localization meets a real deployment. |
| `Securing_WiFi_Fingerprint_Indoor_Localization_2025` | 2025 | The attack side, malicious access points poisoning localization. |
| `ISAC_Fi_Monostatic_Sensing_over_WiFi_2024` | 2024 | Sensing and communication sharing the same waveform on commodity WiFi. |
| `5G_ISAC_2D_Human_Pose_Recognition_2025` | 2025 | The same idea carried over to 5G, useful for the future work section. |

## Group E: Standards and products, not a PDF

These are required by slide 10 and cannot be stored here, so cite them by link.

| Item | What it is | Link |
|---|---|---|
| IEEE Std 802.11bf-2025 | The WLAN Sensing amendment, published 26 September 2025. The single most important citation in the whole report. | https://standards.ieee.org/ieee/802.11bf/11574/ |
| IEEE 802.11 Task Group BF | Working group page with the drafting history and timeline. | https://www.ieee802.org/11/Reports/tgbf_update.htm |
| NIST article on 802.11bf | A readable explanation of the standard from NIST. | https://www.nist.gov/publications/ieee-80211bf-enabling-widespread-adoption-wi-fi-sensing |
| Cognitive Systems, WiFi Motion | Shipping commercial WiFi sensing product. | https://www.cognitivesystems.com |
| Origin AI | Another commercial WiFi sensing vendor. | https://www.originwirelessai.com |
| Linksys Aware | Consumer router feature built on WiFi sensing. | https://www.linksys.com |

## Group F: Datasets and code, cite these as sources

| Item | Link |
|---|---|
| SenseFi benchmark code | https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark |
| Awesome WiFi CSI Sensing, curated paper list | https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing |
| Widar3.0 dataset | https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset |
| Widar3.0 project page | http://tns.thss.tsinghua.edu.cn/widar3.0/ |
| UT-HAR original dataset | https://github.com/ermongroup/Wifi_Activity_Recognition |

## A note on how to use these

Slide 10 says no copied text and no copied figures. Every figure in our report has to be redrawn and every idea has to be restated in our own words. The safe habit is to read a paper, close it, and then write the summary from memory, checking afterwards that we have not accidentally reproduced its phrasing.
