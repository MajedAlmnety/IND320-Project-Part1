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
│   │   └── 4_Extras.py         # Page 4: Extra widgets and charts
├── notebook/
│   └── part1_dashboard_basics.ipynb   # Jupyter Notebook (main development)
├── requirements.txt
└── README.md
```

---

## Streamlit App
Live app: [https://ind320-project-part1-mvjrbnbyefycsijdcn3im4.streamlit.app](https://ind320-project-part1-mvjrbnbyefycsijdcn3im4.streamlit.app)

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

