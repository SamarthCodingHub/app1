import streamlit as st
from streamlit_molstar import st_molstar
import MDAnalysis as md
from openmm.app import *
from openmm import *
from openmm.unit import *
from MDAnalysis.analysis import dihedrals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import requests
import os

# --- MD Simulation Function ---
def run_simulation(pdb_filename, temp, traj_filename):
    pdb = PDBFile(pdb_filename)
    ff = ForceField('amber10.xml')
    system = ff.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic)
    integrator = LangevinIntegrator(temp, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(DCDReporter(traj_filename, 1000))
    simulation.step(int(400*picoseconds / (0.002*picoseconds)))
    return traj_filename

# --- Trajectory Visualization ---
def visualize_trajectory(pdb_file, traj_file):
    st_molstar(pdb_file, traj_file, key="traj", height=600)

# --- End-to-End Distance Analysis ---
def analyze_end_to_end_distance(pdb_file, traj_file):
    sys = md.Universe(pdb_file, traj_file)
    N_terminus = sys.select_atoms('resid 1 and name N')
    C_terminus = sys.select_atoms('resid 25 and name C')
    dist = []
    timesteps = []
    for frame in sys.trajectory:
        timesteps.append(frame.time)
        dist.append(np.linalg.norm(N_terminus.positions - C_terminus.positions))
    data = pd.DataFrame({'Timesteps': timesteps, 'End-to-End Distance': dist})
    fig = px.line(data, x='Timesteps', y='End-to-End Distance', labels={'End-to-End Distance': 'end-to-end distance, Å'})
    st.plotly_chart(fig)

# --- Ramachandran Plot Analysis ---
def analyze_ramachandran_plot(pdb_file, traj_file):
    sys = md.Universe(pdb_file, traj_file)
    ram = dihedrals.Ramachandran(sys).run()
    fig, ax = plt.subplots(figsize=(8, 8))
    ram.plot(ax=ax, color='k', marker='.')
    st.pyplot(fig)

# --- PCA Analysis ---
def perform_pca(pdb_file, traj_file):
    sys = md.Universe(pdb_file, traj_file)
    CA_atoms = sys.select_atoms('name CA')
    N = len(CA_atoms)
    M = len(sys.trajectory)
    X = np.empty((M, int(N * (N - 1) / 2)))
    for k, frame in enumerate(sys.trajectory):
        x = []
        for i in range(len(CA_atoms)):
            for j in range(i+1, len(CA_atoms)):
                d = np.linalg.norm(CA_atoms[i].position - CA_atoms[j].position)
                x.append(d)
        X[k] = np.array(x)
    n_components = min(X.shape)
    skl_PCA = PCA(n_components=n_components).fit(X)
    skl_X_transformed = skl_PCA.transform(X)
    fig, ax = plt.subplots()
    ax.scatter(skl_X_transformed[:, 0], skl_X_transformed[:, 1], c=np.arange(M), cmap='viridis')
    ax.set_xlabel("PC #1")
    ax.set_ylabel("PC #2")
    ax.set_title("PCA of Trajectory")
    st.pyplot(fig)

# --- Main App Logic ---
def main():
    st.title("Protein Molecular Dynamics Simulation App")
    st.info("Upload a PDB file and set simulation parameters. Requires OpenMM and MDAnalysis installed locally.")

    # --- Protein Upload ---
    uploaded_pdb = st.file_uploader("Upload a PDB file", type=["pdb"])
    temp = st.slider("Simulation Temperature (K)", min_value=100, max_value=400, step=50, value=300)
    run_md = st.button("Run MD Simulation")

    if uploaded_pdb and run_md:
        pdb_filename = "input.pdb"
        traj_filename = "traj.dcd"
        with open(pdb_filename, "wb") as f:
            f.write(uploaded_pdb.read())
        with st.spinner("Running MD simulation..."):
            run_simulation(pdb_filename, temp, traj_filename)
        st.success("Simulation complete!")

        # --- Trajectory Visualization ---
        st.header("1. Visualize Trajectory")
        visualize_trajectory(pdb_filename, traj_filename)

        # --- End-to-End Distance ---
        st.header("2. End-to-End Distance")
        analyze_end_to_end_distance(pdb_filename, traj_filename)

        # --- Ramachandran Plot ---
        st.header("3. Ramachandran Plot")
        analyze_ramachandran_plot(pdb_filename, traj_filename)

        # --- PCA ---
        st.header("4. PCA Analysis")
        perform_pca(pdb_filename, traj_filename)

if __name__ == "__main__":
    main()

