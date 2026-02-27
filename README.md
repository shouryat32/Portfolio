# Portfolio
# 👋 Hi, I'm Shourya Thapliyal

**Data Scientist & Software Developer** | Melbourne, Australia

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/shourya-thapliyal-53630b153)
[![Email](https://img.shields.io/badge/Email-shouryat32%40gmail.com-red)](mailto:shouryat32@gmail.com)
[![Phone](https://img.shields.io/badge/Phone-%2B61%20423%20058%20511-green)](tel:+61423058511)

---

## 🚀 About Me

Data Scientist with 2 years of software engineering experience at Tata Consultancy Services, specializing in machine learning, NLP, and AI agents. Recent Master of Data Science graduate from the University of Melbourne (2025) with a proven track record of delivering impactful solutions.

**Key Achievements:**
- 🎯 84% ML model accuracy in Kaggle competition (Top 10%)
- ✅ 95% test automation coverage across banking APIs
- 🔗 15+ API integrations for payment processing systems
- 📊 Published research in Springer ERCICA Volume 1

---

## 🎓 Education

**Master of Data Science** | University of Melbourne | 2023 - 2025
- Coursework: Machine Learning, Big Data Analytics, Statistical Modeling, Cloud Computing
- Industry projects: Social media analytics, mental health data pipelines

**Bachelor of Engineering (Computer Science)** | Nitte Meenakshi Institute | 2016 - 2020
- Specialization: Software Development and Data Structures

---

## 💼 Professional Experience

### Software Engineer | Tata Consultancy Services
**Client: Commonwealth Bank of Australia** | Bangalore, India | 2021 - 2022

- Engineered and deployed 15+ API integrations for banking services
- Reduced API integration time by 30% through reusable frameworks
- Achieved 95% test coverage with Selenium and Python automation
- Built responsive front-end components for NETBANK Preference Centre

---

## 🛠️ Technical Skills

### Languages & Frameworks
![Python](https://img.shields.io/badge/Python-Advanced-3776AB?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-Advanced-276DC3?logo=r&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-Proficient-4479A1?logo=postgresql&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-Proficient-F7DF1E?logo=javascript&logoColor=black)

### Machine Learning & AI
- **Libraries:** scikit-learn, XGBoost, CatBoost, FLAML AutoML, TensorFlow
- **NLP:** Hugging Face Transformers, Agents, MCP
- **Specializations:** Classification, Ensemble Methods, Feature Engineering

### Data Engineering & Visualization
- **Engineering:** PySpark, Delta Lake, Pandas, NumPy, ETL Pipelines, Qualtrics API
- **Visualization:** Power BI, Tableau, Plotly, Matplotlib, Seaborn

### Cloud & DevOps
- **Technologies:** Azure Databricks, Docker, Ansible, NGINX, REST APIs, FastAPI, CI/CD
- **Databases:** PostgreSQL, MySQL, CouchDB, NoSQL

### Tools & Methods
- **Development:** Git, JIRA, Jupyter Notebooks, VS Code
- **Testing:** Selenium, Test Automation
- **Methodologies:** Agile/Scrum, Medallion Architecture

---

## 📚 Featured Projects

### ⚡ Melbourne Electricity Market Analysis
**Personal Project | Mar 2023 - Feb 2026 (data coverage)**

[![GitHub](https://img.shields.io/badge/GitHub-View%20Repo-black?logo=github)](https://github.com/shouryat32/melbourne-electricity-market-analysis)
[![API](https://img.shields.io/badge/Live%20API-energy--models.onrender.com-brightgreen)](https://energy-models.onrender.com/docs)

**Description:**
Started with a simple question — is installing solar panels in Melbourne actually worth it? The answer (yes, every Melbourne zone rates A-tier with a ~6 year payback) opened a much bigger investigation into Victoria's electricity market. Built a full end-to-end data platform covering 3 years of Victorian grid data, 26,156 hours of prices, generation, and weather, with three production ML models deployed via a live API.

**Tech Stack:** `Azure Databricks` `PySpark` `Delta Lake` `XGBoost` `CatBoost` `FLAML` `FastAPI` `Power BI` `AEMO` `Python`

**Key Features:**
- Medallion architecture (Bronze → Silver → Gold) processing 500K+ raw records into analytics-ready tables
- Live data pipeline running hourly via Databricks scheduler — AEMO prices + BOM weather
- Nightly silver layer build incrementally appending new data at 2am
- Three ML models: price prediction (R² 0.89), demand forecasting (R² 0.98), spike classifier (ROC-AUC 0.99)
- All models deployed as a live REST API on Render
- Power BI dashboard with 4 pages — Price Intelligence, Generation & Renewables, Weather Correlation, AI Price Intelligence

**Key Findings:**
- 🕛 Cheapest hour: 12pm — $1.40/MWh average
- 🕕 Most expensive hour: 6pm — $177.76/MWh (a $176/MWh daily spread)
- ⚡ Negative pricing (grid oversupply) occurred in 23% of all hours over 3 years
- 🌡️ Summer paradox — Victoria's hottest days actually produce *less* renewable energy (29% vs 55% on cool days) because high-pressure systems kill wind speed

---

### 🏥 Phoenix Australia Mental Health Analytics Pipeline
**Oct 2024 - Dec 2025**

**Description:**
Built an automated data pipeline integrating Qualtrics API to analyze mental health course feedback and track program effectiveness.

**Tech Stack:** `Python` `Power BI` `SQL` `Qualtrics API` `ETL` `Plotly` `Matplotlib`

**Key Features:**
- Automated data collection from Qualtrics surveys
- Interactive dashboards tracking participant outcomes and engagement metrics
- ETL workflows processing participant data for research evaluation
- Real-time analytics for program coordinators

**Impact:**
- Automated reporting reduced manual work by XX%
- Enabled data-driven decisions for XX program participants
- Improved feedback response time by XX%

---

### 🤖 Machine-Generated Text Detection
**Kaggle Competition | Jan 2024 - Mar 2024**

**Description:**
Developed an ensemble machine learning system to detect AI-generated text, achieving 84% accuracy and ranking in the top 10% of participants.

**Tech Stack:** `Python` `scikit-learn` `XGBoost` `Random Forest` `Neural Networks` `NLP`

**Key Techniques:**
- Ensemble methods combining multiple classifiers
- Feature engineering with TF-IDF and word embeddings
- Hyperparameter tuning with cross-validation
- Text preprocessing and tokenization

**Results:**
- ⭐ 84% accuracy on test set
- 🏆 Top 10% ranking among XX participants
- 📈 Improved baseline model by XX%

---

### 🌏 Australia Social Media Analytics Platform
**University of Melbourne | Aug 2024 - Nov 2024**

**Description:**
Scalable cloud-based analytics system investigating the relationship between social media sentiment and crime rates across Victorian Local Government Areas (LGAs). Processed 61GB Twitter dataset and 100,000+ monthly posts to provide actionable insights for law enforcement and policy-making.

**Tech Stack:** `Docker` `CouchDB` `NGINX` `Python` `Ansible` `NLTK` `TextBlob` `Folium` `Geopandas` `Melbourne Research Cloud`

**My Core Contributions:**
- **Cloud Architecture:** Designed and deployed 3-instance distributed system with fault-tolerant infrastructure
- **NoSQL Database:** Built 3-node CouchDB cluster with automatic replication and high availability
- **Big Data Processing:** Developed Python scripts to process 61GB Twitter dataset with MapReduce aggregation
- **NLP Pipeline:** Implemented sentiment analysis using NLTK, TextBlob, and TF-IDF for 100,000+ posts
- **Geospatial Visualization:** Created interactive Folium maps and crime density heatmaps
- **DevOps Automation:** Utilized Ansible for infrastructure-as-code, reducing setup time by 70%

**Key Achievements:**
- ✅ Processed 100,000+ social media posts monthly with distributed architecture
- 🚀 Reduced infrastructure setup time by 70% through Ansible automation
- 📊 Discovered significant correlation between negative sentiment and elevated crime rates
- 🗺️ Mapped sentiment analysis to geographic regions for policy insights
- 🔄 Achieved fault-tolerant storage with automatic CouchDB failover capabilities

**Team:** 3-person collaborative project (Keshav Prasath, Solmaz Maabi, Shourya Thapliyal)

---

### ❤️ Cardiac Arrhythmia Prediction System
**Published Research | Undergraduate Thesis | 2019 - 2020**

**Description:**
Machine learning-based diagnostic system to predict cardiac arrhythmias from electrocardiogram (ECG) data. Published research project that classifies patients into 13 different cardiac conditions, enabling early detection of potentially life-threatening heart rhythm abnormalities.

**Tech Stack:** `Python` `scikit-learn` `NumPy` `Pandas` `Matplotlib` `UCI Dataset`

**Model Performance:**
- Achieved **72%+ accuracy** on UCI Cardiac Arrhythmia dataset
- Systematic comparison of 8 different model configurations
- End-to-end ML pipeline from raw data to predictions

**Research Impact:**
- 📝 **Published in Springer ERCICA Volume 1** (2020)
- 🎤 Presented at International Conference on Emerging Research in Computing, Information, Communication and Applications
- 🏥 Contributes to AI-assisted medical diagnostics for scalable healthcare

---

## 📝 Publications

**Thapliyal, S. et al.** "Cardiac Arrhythmia Prediction Using Machine Learning"
- *Springer ERCICA Volume 1*
- International Conference on Emerging Research in Computing, Information, Communication and Applications (2020)
- [Read Paper](https://link.springer.com/chapter/10.1007/978-981-16-1342-5_3)

---

## 🏆 Certifications

- **Hugging Face:** AI Agents Fundamentals, MCP, Automation in Production (2025)
- **Google Data Analytics Professional Certificate** (2025)
- **Intermediate R** - DataCamp (2023)
- **SQL Basic and Intermediate Certifications** - HackerRank (2025)

---

## 📫 Let's Connect!

I'm always interested in collaborating on data science projects, particularly in:
- 🤖 Machine Learning & AI Applications
- 📊 Data Analytics & Visualization
- 🏥 Healthcare Analytics
- 💰 Financial Services Solutions

Feel free to reach out:
- 📧 Email: shouryat32@gmail.com
- 💼 LinkedIn: [www.linkedin.com/in/shourya-thapliyal-53630b153](https://www.linkedin.com/in/shourya-thapliyal-53630b153)
- 📱 Phone: +61 423 058 511
