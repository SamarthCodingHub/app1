import streamlit as st
import pandas as pd
import py3Dmol

st.title("Protein Structure Viewer")
st.write("Enter a PDB ID to view the protein structure.")

pdb_id = st.text_input("Enter PDB ID:", "5BRO")

def fetch_pdb(pdb_id):
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    return pdb_url

pdb_url = fetch_pdb(pdb_id)

def render_mol(pdb_url):
    view = py3Dmol.view(width=400, height=400)
    view.addModel(requests.get(pdb_url).text, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return view.show()

st.write(render_mol(pdb_url))


