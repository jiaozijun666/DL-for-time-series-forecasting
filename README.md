# DL-for-time-series-forecasting
作业要求：

Deep Learning for Time-Series Forecasting
DSS5104
Overview
The goal of this assignment is to critically explore and evaluate recent deep learning methods for time-series prediction. While classical models like ARIMA, Prophet, and gradient boosting have long been standard for temporal forecasting, deep learning architectures, particularly those designed for sequential modeling, Deep Learning models have shown increasing promise, althoug it is still not clear if they are the best choice for all time-series problems.

In this assignment, you will evaluate the performance, scalability, and robustness of a selection of modern deep learning methods using publicly available implementations on a variety of real-world time-series forecasting tasks. You are not expected to implement models from scratch but must demonstrate an understanding of their proper application.

Assignment Objectives
Understand and apply recent deep learning models for time-series forecasting
Benchmark their performance on multiple datasets
Compare them against classical baselines that do not use deep learning
Analyze the advantages, limitations, and practical considerations of deep models in time-series settings
Reflect on when and why deep models outperform traditional methods—or don’t
Datasets and Prediction Tasks
You must choose at least five time-series datasets. These should:

Involve forecasting over a meaningful horizon (multi-step if possible)
Come from diverse domains (e.g., energy, finance, climate, retail, transportation)
Be publicly available and reasonably complex
Key Questions to Address
Performance: How do deep time-series models compare to classical models in accuracy? On which kinds of data do they excel or struggle? Are they really better than classical models?
Scalability: How do they perform in terms of training time, memory usage, etc…?
Robustness: How sensitive are models to data volume, missing values, input noise, and hyperparameters?
Generalization: Do deep models overfit, especially on smaller datasets or with limited history?
Practicality: Would you recommend deep models in real-world time-series applications? Under what conditions?
Deliverables
Written Report (PDF)
Max 10 pages (shorter is welcome if well-written and to the point)
Focus on insights, comparisons, and analysis: no code in the report
Code Submission
Organized scripts or notebooks (via GitHub or zip) for:
Data preprocessing and transformation
Model training, evaluation, and tuning
Metrics reporting and visualizations
Code must be clear, well-documented, and reproducible
Final Thoughts
This assignment is your opportunity to engage with the cutting edge of time-series forecasting using deep learning. These models bring new capabilities but also new challenges. Through hands-on experimentation and thoughtful evaluation, you’ll build the ability to choose, justify, and defend modeling decisions in time-series contexts.

By the end, you should be able to answer:

Are deep models worth the overhead in time-series problems? It is fine if you find that they are not!
When are they most beneficial?
What trade-offs do they introduce compared to classical approaches?
