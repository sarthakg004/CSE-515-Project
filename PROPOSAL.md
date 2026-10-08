# Project Proposal

**Course:** CSE515, Advanced Communications Networks and Systems
**Instructor:** Dr. Ajay Pratap
**Title:** WiFi Sensing with Channel State Information for Device Free Human Activity Recognition and Indoor Localization
**Team members:** Saarthak Gupta (22134024), Shreya Sangma (22134025), Vaibhav Jain (22164020)

## What the project is about

Every WiFi device already measures how the radio signal changes between a transmitter and a receiver. This measurement is called Channel State Information, or CSI. When a person walks, sits, falls, or simply breathes inside a room, their body reflects and blocks the radio waves, and the CSI values change in a pattern that is specific to that movement. This means an ordinary WiFi link can be used as a sensor. Nobody has to wear a device, no camera is needed, and the sensing works in the dark and through walls.

Our project builds a complete software system that takes real CSI recordings from public datasets, processes them, and trains deep learning models to answer two questions. The first question is what activity a person is performing. The second question is where that person is located inside the room. We then measure how accurate, how fast, and how large these models are, and we compare them against each other in a fair and repeatable way.

## Why this topic is worth doing now

This is not a settled textbook topic, it is an active area that the industry is standardising right now. The IEEE published the 802.11bf standard for WLAN Sensing in September 2025, which defines how WiFi devices should discover sensing support, set up a sensing session, exchange measurements, and report the results. Companies such as Cognitive Systems, Origin AI, and Linksys already ship motion sensing features built on this idea, and consumer routers with standardised sensing support are expected in volume during 2026. At the same time, the research community has not solved the hardest problem in the field, which is that a model trained in one room, with one person, and with one hardware setup usually performs badly when it is moved to a new room or a new person. This gap between a working demonstration and a deployable system is exactly what our project will study.

## The gap we have already found

We began by examining how the standard SenseFi benchmark divides the Widar3.0 gesture dataset into a training half and a testing half, and we found that the division is not a fair one. Every one of the seventeen volunteers appears on both sides. So do all eight standing positions and all five body orientations. The model is therefore tested on people it has already trained on, standing where it has already seen them stand. The published accuracy is an in domain number, and it does not answer the question that matters, which is whether the system works for a person it has never met, standing somewhere it has never seen. Widar3.0 records the volunteer, the position, and the orientation of every sample in its filenames, so a correct split can be built, and to our knowledge it has not been.

## What we will actually build

We start from the SenseFi benchmark, which provides a clean PyTorch implementation of eleven model families ranging from a simple multilayer perceptron to a Vision Transformer, and we reproduce its baseline results on the UT-HAR and NTU-Fi datasets. To those results we add parameter count and inference time, which the benchmark never reports although they decide whether a model could run on real hardware. We then rebuild the Widar3.0 split so that whole conditions are held out, training on some volunteers and testing on volunteers the model has never seen, and likewise for body orientation. We measure how far accuracy falls, and that fall is our main result. We then try to recover it using data augmentation and self supervised pre training, and we report the outcome honestly, including any technique that fails to help. Finally we check whether a compressed model retains most of its accuracy, since sensing that is too expensive to compute will never run on a real access point.

## Data, tools, and effort

All four datasets, namely UT-HAR, NTU-Fi HAR, NTU-Fi HumanID, and Widar3.0, are publicly released by their authors and are already downloaded into our repository. The entire project is software only and requires Python, PyTorch, and NumPy. No hardware needs to be built or purchased, and no data collection campaign is needed. The literature survey will cover technical papers from 2024 onwards, the IEEE 802.11bf standard document, and the commercial products listed above, so that the report reflects research, standards, and industry practice together. Every figure in the final report will be drawn by us and every idea will be described in our own words.

## Deliverables

For the midterm submission we will deliver the literature survey, the reproduced baselines compared by accuracy, model size, and inference time, and the evidence for the split problem described above. For the end term submission we will deliver the cross domain results on the rebuilt Widar3.0 splits, the measured effect of each recovery technique, the efficiency study, and the full written report.

## References

1. J. Yang, X. Chen, D. Wang, H. Zou, C. X. Lu, S. Sun, and L. Xie, "SenseFi: A Library and Benchmark on Deep Learning Empowered WiFi Human Sensing," Patterns, Cell Press, vol. 4, no. 3, 2023. Available at https://arxiv.org/abs/2207.07859
2. IEEE Standards Association, "IEEE Std 802.11bf-2025, Amendment 4: Enhancements for Wireless LAN Sensing," published 26 September 2025. Available at https://standards.ieee.org/ieee/802.11bf/11574/
3. F. Meneghello, C. Chen, C. Cordeiro, and F. Restuccia, "Toward Integrated Sensing and Communications in IEEE 802.11bf Wi-Fi Networks," IEEE Communications Magazine, 2023. Available at https://arxiv.org/abs/2212.13930
4. Y. Ma, G. Zhou, and S. Wang, "A Survey of Commodity WiFi Sensing in 10 Years: Current Status, Challenges, and Opportunities," 2021. Available at https://arxiv.org/abs/2111.07038
5. J. Yang, X. Chen, H. Zou, D. Wang, Q. Xu, and L. Xie, "EfficientFi: Towards Large Scale Lightweight WiFi Sensing via CSI Compression," IEEE Internet of Things Journal, 2022. Available at https://arxiv.org/abs/2204.04138
6. J. Strohmayer and M. Kampel, "Data Augmentation Techniques for Cross Domain WiFi CSI Based Human Activity Recognition," 2024. Available at https://arxiv.org/abs/2401.00964
7. J. Strohmayer and M. Kampel, "WiFlexFormer: Efficient WiFi Based Person Centric Sensing," 2024. Available at https://arxiv.org/abs/2411.04224
8. B. Barahimi, H. Tabassum, M. Omer, and O. Waqar, "Lessons from Deploying Learning Based CSI Localization on a Large Scale ISAC Platform," 2025. Available at https://arxiv.org/abs/2504.17173
9. Y. Zheng, Y. Zhang, K. Qian, G. Zhang, Y. Liu, C. Wu, and Z. Yang, "Widar3.0: WiFi Based Activity Recognition Dataset," IEEE DataPort. Available at https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
10. Cognitive Systems Corporation, "WiFi Motion," commercial WiFi sensing product. Available at https://www.cognitivesystems.com
