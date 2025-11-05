# Home page for the Streamlit app
# - Sets global page config
# - Provides quick instructions and link placeholders

import streamlit as st

st.set_page_config(page_title="IND320 – Part 1", page_icon="📊", layout="wide")

st.title("IND320 – Project Work: Part 1 (Dashboard basics)")
st.write(
    "Use the sidebar to navigate to the other pages. "
    "Streamlit automatically shows any scripts in the `pages/` folder as separate pages."
)

st.markdown("### Links")
st.markdown("- GitHub: *..*")
st.markdown("- Streamlit App: *..*")



st.markdown("---")
st.markdown(
    """
    **What you’ll find in this app:**
    - Page 2 shows a table with a per-variable sparkline for the **first month**.
    - Page 3 provides a full plotting view with **monthly/weekly aggregation**.
    - Includes **scaling options** so you can plot all columns together naturally even when their scales differ.
    """
)


