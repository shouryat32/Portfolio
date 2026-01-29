# Australia Social Media Analytics on the Cloud

## Project Overview
A scalable cloud-based analytics system investigating the relationship between social media sentiment and crime rates across Local Government Areas (LGAs) in Victoria, Australia. This project leveraged big data technologies, containerization, and advanced NLP techniques to process and analyze 100,000+ social media posts monthly.

**Duration:** August 2024 - November 2024  
**Team:** 3-person collaborative project (Keshav Prasath, Solmaz Maabi, Shourya Thapliyal)  
**Institution:** University of Melbourne - School of Computing and Information Systems

---

## Problem Statement
Understanding the correlation between public sentiment expressed on social media platforms and real-world crime statistics to provide actionable insights for law enforcement and policy-making. The challenge involved processing massive volumes of unstructured social media data (61GB Twitter dataset), correlating it with crime statistics across multiple geographic regions, and presenting findings through an interactive platform.

---

## My Technical Contributions

### 1. **Cloud Infrastructure Architecture & Deployment**
- Designed and deployed a **3-instance distributed system** on University of Melbourne Research Cloud
  - **Instance 1 (Data Management):** 3-node CouchDB cluster with 100GB storage for high-availability data storage
  - **Instance 2 (Analysis):** Jupyter notebook environment with Python analytics stack (60GB storage)
  - **Instance 3 (Frontend):** NGINX web server for visualization platform (100GB storage)
- Implemented **Docker containerization** across all instances for consistency and reproducibility
- Utilized **Ansible** for infrastructure-as-code (IaC) automation and configuration management
- Achieved **70% reduction in infrastructure setup time** through automated deployment scripts

### 2. **Big Data Processing & Storage**
- Built **3-node CouchDB NoSQL cluster** providing:
  - High availability with fault tolerance (survives 1-2 node failures)
  - Automatic data replication across all nodes
  - Efficient JSON-based document storage for 100,000+ monthly posts
  - MapReduce capabilities for distributed data aggregation
- Developed **Python data harvesting scripts** to extract and filter relevant features from 61GB Twitter dataset
- Implemented **Mastodon API integration** with dynamic credential management in CouchDB
- Processed tweets from 2020-2022, filtering by location (Victoria), language, and relevance

### 3. **Natural Language Processing & Sentiment Analysis**
- Implemented **sentiment analysis pipeline** using:
  - **NLTK & TextBlob** for natural language processing
  - **TF-IDF and word embeddings** for feature extraction
  - Custom algorithms accounting for sarcasm, slang, emojis, and multilingual content
- Developed **MapReduce algorithms** in CouchDB for:
  - Counting tweet distribution by location
  - Calculating hashtag frequencies
  - Aggregating sentiment scores by LGA
- Processed and analyzed **geospatial data** mapping tweets to specific Local Government Areas

### 4. **Data Visualization & Web Platform**
- Created **interactive visualizations** using:
  - **Folium maps** for geospatial sentiment distribution across Victoria
  - **Matplotlib & Seaborn** for sentiment/subjectivity histograms
  - **Geopandas** for crime density heatmaps
  - **WordCloud** for trending topic analysis
- Developed **NGINX-hosted web application** featuring:
  - Interactive Folium maps showing sentiment, subjectivity, and word count by LGA
  - Crime division statistics heatmaps
  - Geospatial crime density visualizations
- Implemented **automated data transfer scripts** from Analysis to Frontend instance

### 5. **System Integration & DevOps**
- Orchestrated **Docker Compose** multi-container applications
- Implemented **automated volume mounting** for persistent storage
- Designed **error handling mechanisms** including:
  - Exception handling for API connection failures
  - Majority voting for database inconsistency resolution
  - Automatic failover between CouchDB nodes
- Applied **security best practices**:
  - Key-based authentication for cloud instances
  - Container isolation following principle of least privilege
  - CouchDB authentication and authorization controls
  - Input validation to prevent injection attacks

---

## Technical Stack

**Cloud & Infrastructure:**
- Melbourne Research Cloud (3 instances: 2VPC 100GB, 2VPC 60GB, 2VPC 100GB)
- Docker & Docker Compose
- Ansible (Infrastructure as Code)

**Data Storage & Processing:**
- CouchDB (3-node NoSQL cluster)
- Python (Pandas, NumPy)
- MapReduce for distributed aggregation

**Natural Language Processing:**
- NLTK (Natural Language Toolkit)
- TextBlob (Sentiment Analysis)
- FuzzyWuzzy (Text Matching)

**Data Visualization:**
- Matplotlib & Seaborn
- Plotly
- Folium (Interactive Maps)
- Geopandas (Geospatial Visualization)
- WordCloud

**Web Technologies:**
- NGINX Web Server
- Selenium WebDriver
- Jupyter Notebook

**Data Sources:**
- Twitter (61GB JSON dataset, 2020-2022)
- Mastodon API
- SUDO Crime Statistics (Victoria, 2011-2019)
- Geospatial Data (OpenDataSoft)

---

## Key Achievements & Insights

### Technical Achievements:
- **Processed 100,000+ social media posts monthly** with 3-node distributed architecture
- **Reduced infrastructure setup time by 70%** through Ansible automation
- **Achieved fault-tolerant storage** with automatic failover capabilities
- Successfully **integrated 4 diverse data sources** (Twitter, Mastodon, crime stats, geospatial data)
- Built **scalable system architecture** capable of handling 61GB+ datasets

### Analytical Insights:
- **Discovered significant correlation** between negative Twitter sentiment and elevated crime rates in specific LGAs
- Identified **temporal patterns** in both social media sentiment and crime occurrences
- Found **Property & Deception offenses (Division B)** accounted for the largest proportion of crimes
- Observed **higher negative sentiment in Melbourne CBD**, correlating with higher crime density
- Detected **geographic clusters** of suburbs with both hostile sentiment and increased crime rates

### Data Findings:
- Sentiment distribution: Predominantly neutral with slightly more positive than negative sentiments
- Subjectivity was polarized, clustering at both extremes
- Crime concentration highest in inner city and certain eastern suburbs
- Successfully mapped sentiment analysis to geographic regions for policy insights

---

## Challenges & Solutions

### 1. **Network Latency on Research Cloud**
- **Challenge:** Sporadic network delays hindering data processing efficiency
- **Solution:** Strategically scheduled tasks during off-peak hours; implemented monitoring and resource allocation adjustments

### 2. **Processing 61GB Twitter Dataset**
- **Challenge:** Memory management and efficient extraction from massive JSON file
- **Solution:** Developed streaming data extraction scripts with targeted filtering; utilized CouchDB's efficient JSON storage

### 3. **Ansible Deployment Complexity**
- **Challenge:** Playbook execution errors and role dependency issues
- **Solution:** Pivoted to manual instance creation with embedded scripts while maintaining automation for repetitive tasks

### 4. **Sentiment Analysis Accuracy**
- **Challenge:** Detecting sarcasm, slang, emojis, and multilingual content
- **Solution:** Implemented multi-language NLP tools; refined algorithms to account for context and language nuances

### 5. **Visualization of Large Datasets**
- **Challenge:** Folium and Selenium tools crashing with significant data volumes
- **Solution:** Implemented data aggregation at LGA level; optimized rendering with selective data display

---

## System Architecture

```
Melbourne Research Cloud
│
├── INSTANCE 1 (Data Management - 100GB)
│   ├── CouchDB Node 1 (Docker Container)
│   ├── CouchDB Node 2 (Docker Container)
│   ├── CouchDB Node 3 (Docker Container)
│   └── Mastodon Harvester Script
│
├── INSTANCE 2 (Analysis - 60GB)
│   ├── Jupyter Notebook (Docker Container)
│   ├── Python Analytics Stack
│   │   ├── Pandas, NumPy
│   │   ├── NLTK, TextBlob
│   │   ├── Matplotlib, Folium
│   │   └── Geopandas
│   └── Analysis Results Transfer Script
│
└── INSTANCE 3 (Frontend - 100GB)
    ├── NGINX Web Server (Docker Container)
    └── Interactive Visualization Platform
```

**Data Flow:**
1. Twitter/Mastodon data → CouchDB Cluster (Instance 1)
2. Analysis Instance queries CouchDB → Performs sentiment analysis
3. Generates visualizations → Stores in mounted volume
4. Transfer script → Moves visualizations to Frontend Instance
5. NGINX serves interactive web platform to users

---

## Impact & Applications

### For Law Enforcement & Policy:
- Provides early warning system for potential crime hotspots based on sentiment monitoring
- Enables data-driven resource allocation across different LGAs
- Offers geographic visualization of public mood and crime correlation

### For Research:
- Demonstrates scalability of container-based big data analytics
- Showcases effective integration of diverse data sources
- Validates social media as indicator of real-world societal trends

### For Business Applications:
- Framework applicable to customer sentiment analysis
- Scalable architecture adaptable to various domain contexts
- Proof of concept for cloud-native analytics platforms

---

## Future Enhancements

1. **Causal Analysis:** Investigate causal mechanisms underlying sentiment-crime correlation beyond correlation
2. **Enhanced NLP:** Implement machine learning models trained on broader sentiment indicators
3. **Real-time Processing:** Migrate from historical data to real-time Twitter/social media streams
4. **Multi-language Support:** Expand analysis beyond English-language tweets
5. **Finer Granularity:** Implement suburb-level analysis instead of LGA-level for more detailed insights
6. **Advanced Visualization:** Explore big data visualization tools designed for large-scale datasets
7. **AI Integration:** Leverage AI for automated infrastructure management and enhanced predictions

---

## Key Learnings

- **Cloud Architecture:** Hands-on experience designing scalable, fault-tolerant distributed systems
- **Big Data Processing:** Mastered handling and analyzing massive unstructured datasets efficiently
- **DevOps Practices:** Gained expertise in containerization, infrastructure-as-code, and automated deployment
- **NLP Challenges:** Developed strategies for handling nuanced human language in social media context
- **System Resilience:** Implemented robust error handling and failover mechanisms for production systems
- **Collaborative Development:** Successfully coordinated technical contributions across 3-person team

---

## Project Documentation

- **Full Report:** [Available upon request]
- **System Demo:** Deployed web interface at http://172.26.132.47:8080/
- **Code Repository:** GitLab (Team 58)

---

## References & Related Work

This project builds upon research in:
- Social media sentiment analysis and crime correlation studies
- MapReduce for distributed data processing
- Docker containerization for reproducible research
- NoSQL database security and sharding
- Ansible automation frameworks
- Cloud computing latency optimization

---

*This project demonstrates proficiency in cloud architecture, big data analytics, NLP, DevOps practices, and end-to-end system design for real-world social impact applications.*
