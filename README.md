## 任务：
1. data+清理 （3/5）
2. 原始模型选择 (ARIMA, XGBoost, Prophet......)
3. modern deep learning model选择（LSTM，Transformer，GRU，TFT, GRU.......)
4. 做对比，comparison criteria在googledoc有详细版）
5. 做分析，写报告

## 📌 Overview

The goal of this assignment is to critically explore and evaluate **recent deep learning methods** for time-series prediction. While classical models like **ARIMA**, **Prophet**, and **gradient boosting** have long been standard, deep learning architectures—especially those designed for sequential modeling—have shown increasing promise. However, it's still not clear if they are the best choice for all time-series problems.

In this assignment, you will:

- Evaluate the **performance, scalability, and robustness** of modern deep learning methods
- Use **public implementations** on real-world time-series forecasting tasks
- Demonstrate **understanding of model application** (not implementation from scratch)

---

## 🎯 Assignment Objectives

- Understand and apply recent **deep learning models** for time-series forecasting
- Benchmark their performance on **multiple datasets**
- Compare them against **classical baseline models**
- Analyze advantages, limitations, and practical considerations
- Reflect on **when and why** deep models outperform traditional methods — or don't

---

## 📊 Datasets and Prediction Tasks

You must select **at least five time-series datasets** that:

- Involve **forecasting over a meaningful horizon** (preferably multi-step)
- Come from **diverse domains** (e.g., energy, finance, climate, retail, transportation)
- Are **publicly available** and of **reasonable complexity**

---

## 🧠 Key Questions to Address

1. **Performance**  
   How do deep time-series models compare to classical models in accuracy?  
   On which kinds of data do they excel or struggle?  
   Are they really better than classical models?

2. **Scalability**  
   How do they perform in terms of training time, memory usage, etc.?

3. **Robustness**  
   How sensitive are models to data volume, missing values, input noise, and hyperparameters?

4. **Generalization**  
   Do deep models overfit, especially on smaller datasets or with limited history?

5. **Practicality**  
   Would you recommend deep models in real-world time-series applications?  
   Under what conditions?

---

## 📄 Deliverables

### 1. Written Report (PDF)
- **Maximum 10 pages** (shorter is fine if well-written and concise)
- Focus on **insights, comparisons, and analysis**
- **Do not include code** in the report

### 2. Code Submission (GitHub repo or ZIP)
Organized scripts or notebooks for:
- Data preprocessing and transformation
- Model training, evaluation, and tuning
- Metric reporting and visualizations

Your code must be:
- **Clear**, **well-documented**, and **reproducible**

---

## 🧭 Final Thoughts

This assignment is your opportunity to engage with the **cutting edge of time-series forecasting** using deep learning. These models bring **new capabilities** but also **new challenges**.

Through **hands-on experimentation and thoughtful evaluation**, you’ll build the ability to choose, justify, and defend modeling decisions in real-world time-series applications.

By the end, you should be able to confidently answer:

- Are deep models worth the overhead in time-series problems?
- When are they most beneficial?
- What trade-offs do they introduce compared to classical approaches?

---
