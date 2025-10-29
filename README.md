# IND320 – Project Part 1: Dashboard Basics

This repository contains my submission for **IND320 – Dashboard Basics (Part 1)**.  
It includes both the Jupyter Notebook used for development and documentation, and the Streamlit app used to present the results interactively.

---

## Repository Structure
```
IND320-Project-Part1/
├── app/
│   ├── Home.py                 # Streamlit front page
│   ├── open-meteo-subset.csv   # Local dataset used by the app
│   ├── pages/
│   │   ├── 2_Table.py          # Page 2: Data table with sparklines
│   │   ├── 3_Plots.py          # Page 3: Plots with filters
│   │   └── 4_Production_2021.py        
        └── 5_Extras.py      # Page 4: Extra widgets and charts
├── notebook/
│   └── part1_dashboard_basics.ipynb   # Jupyter Notebook (main development)
    └── project_work_part2.ipybn 
├── requirements.txt
└── README.md
```

---

## Streamlit App
Live app: [https://ind320-project-part1-mvjrbnbyefycsijdcn3im4.streamlit.app]

This app is deployed automatically from this repository via Streamlit Cloud.  
Any changes pushed to GitHub are reflected in the live app.

---

## Jupyter Notebook
The notebook (`notebook/part1_dashboard_basics.ipynb`) contains:
- Data loading and exploration with pandas.
- Summary statistics and initial plots.
- Visualizations for each column and normalized combined plots.
- A **General** section with:
    - Short description of AI usage.
    - A 300–500 word log describing the work process.

The notebook must be fully run and then exported to PDF before submission, so that outputs and plots are visible.  
Both the `.ipynb` and the PDF export are part of this submission.

---

## AI Usage
I used AI tools (ChatGPT) in the following ways:
- To scaffold the Notebook and Streamlit app.
- To suggest plotting techniques and code examples.
- To help fix path issues and implement caching in Streamlit.
- To generate Git commands and fix push errors when working with GitHub.

All suggestions were tested and adapted manually to ensure correctness.

---

## Project Log
The detailed 300–500 word log is inside the Jupyter Notebook (Section 6).  
It covers both Jupyter Notebook and Streamlit development, challenges faced, and lessons learned.

---

## Running Locally
Clone this repository and install requirements:

```bash
git clone https://github.com/MajedAlmnety/IND320-Project-Part1.git
cd IND320-Project-Part1
pip install -r requirements.txt
streamlit run app/Home.py
```

---

# Project Work Part2 – Data Analysis and Streamlit Dashboard

## Overview
This project demonstrates a complete data workflow using **Jupyter Notebook** for data analysis and **Streamlit** for web-based visualization.  
The goal of the project is to prepare and process a dataset, store it in a **MongoDB** database, and build an interactive dashboard that allows users to explore and visualize the data easily.  

The work focuses on combining data engineering, analysis, and visualization into one integrated process.  
Through this project, I gained experience in working with Python libraries, managing databases, and building user-friendly interfaces for data exploration.

---

## Main Components

### 1. **Jupyter Notebook**
- Used for data cleaning, transformation, and analysis.  
- Libraries used: `pandas`, `numpy`, `pyspark`, and `pymongo`.  
- Connected to a **MongoDB** database to store processed data.  
- Includes clear comments and Markdown explanations for each step.

### 2. **MongoDB**
- Acts as the storage system for processed data.  
- Allows for structured and efficient data retrieval for analysis and visualization.  

### 3. **Streamlit Application**
- Used to build an interactive dashboard for data visualization.  
- Integrates charts and graphs built with `matplotlib` and `plotly`.  
- Users can explore, filter, and interact with the dataset through a simple web interface.  

---

## Setup and Installation

1. Clone or download the project folder.  
2. Install the required dependencies:
   ```bash
   pip install pandas numpy pyspark pymongo streamlit matplotlib plotly python-dotenv
