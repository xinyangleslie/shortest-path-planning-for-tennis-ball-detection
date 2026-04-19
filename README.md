# Shortest Path Planning For Tennis Ball Detection





## Requirement

Upload your final project materials in this folder.

Your submission must include **both** the written report and all accompanying scripts/documentation needed to fully reproduce your experiments.

### **1. Final Project Report (IEEE 2-Column Format)**

Submit your report as a **PDF** generated from the IEEE two-column LaTeX template I have provided.

Your paper **must** include the following sections:

- **Title**
- **Authors**
- **Abstract -** A concise summary of the problem, approach, and key findings.

- **Introduction -** Include:

  - A clear **problem statement**,

  - A brief overview of **related work**,

  - A short description of the **specific contribution** your paper makes.

- **Methodology -** Describe your approach **in full detail**, including:

  - Model architecture or algorithm used,

  - Training strategy/setup,

  - Any preprocessing or augmentation,

  - Implementation details and **justification for design choices**.

- **Experimental Results -** This section must provide:

  - A detailed explanation of the **experimental setup**,

  - **Dataset description** and how it was used,

  - **Hardware and software** on which you ran your experiments,

  - Presentation of the **results** (tables, plots, metrics),

  - **Discussion** of your findings, including:
    - Cases where your method performed well,
    - Cases where it failed or underperformed, and **why**.

- **Conclusion -** Summarize your findings, limitations, and possible future work

- **Bibliography**

### **2. Code & Reproducibility Materials (ZIP File)**

Upload a **ZIP file** containing:

- All **scripts** used for training, testing, and evaluation,
- A **README file** with **clear documentation** describing:
- How to install required libraries,
- How to run each script step-by-step,
- Any configuration files or parameters needed,
- Expected outputs at each stage.

Your documentation should be sufficiently clear for another person to **replicate your experiments** without additional clarification.

### **3. Dataset Access**

Include in your submission:

- A **link or citation** to the dataset(s) used,
- Any notes on preprocessing, custom splits, or special instructions.

### Extra Credit Opportunity

Students who provide a **fully documented Git repository** will receive **extra credit**.

To qualify, the repository must include:

- Clean, well-organized code structure
- Clear commit history reflecting development progress
- A comprehensive README (setup, usage, and experiment reproduction)
- Proper documentation/comments within the code
- Instructions that allow a third party to reproduce results directly from the repository

Providing a public repository (e.g., GitHub) is encouraged, but a private repository with shared access is also acceptable.

### **Submission Notes**

- Ensure that all files upload correctly (PDF + ZIP).
- Make sure the PDF is generated cleanly from LaTeX and follows the IEEE two-column format.



## Abstract

这个项目我们主要做的是：

1. 自己采集网球数据集（Rainbow网站+Court1(outdoor)+Court1(indoor)），训练对比YOLO26的多种模型
2. 基于最优的模型训练效果，来结合cv算法，实现tracking，并显示在Rviz2
3. 后面使用最短路径寻路的算法来实现模拟机器人捡球







