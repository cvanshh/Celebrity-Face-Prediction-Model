import streamlit as st

st.title("Streamlit Test App")

st.write("If you can see this, Streamlit is working.")

name = st.text_input("Enter your name")

if name:
    st.success(f"Hello {name}, everything is working fine 🚀")

num = st.slider("Pick a number", 0, 100)

st.write("Square:", num * num)