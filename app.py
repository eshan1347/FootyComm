import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from transformers import GPT2Tokenizer 
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any, Callable
from pred import *
from load_data import *

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = Encoder(h=64,n=2, e=64, a=4, o=64).to(device)
    decoder = Decoder(h=64,n=2, e=64, a=4, o=50257).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    checkpoint = torch.load('./seq2seq_checkpoint.pt', weights_only=True, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    st.title("Footy Commentary Generator")
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    # Tab selection
    tab_selection = st.sidebar.radio(
        "Select Input Method:",
        ["Random Sample from Test Set", "Custom Input"]
    )
    # Decoding configuration section
    st.sidebar.header("Decoding Configuration")
    st.session_state.decoding_mode = st.sidebar.selectbox(
        "Decoding Mode",
        ["greedy", "sample", "top-k", "diverse-beam-search", "min-bayes-risk"]
    )
    # Parameters based on decoding mode
    st.session_state.decoding_params = {}
    st.session_state.decoding_params['max_len'] = st.sidebar.slider('Max length', 1, 500, 50)
    st.session_state.decoding_params['temperature'] = st.sidebar.slider('Temperature', 0.0, 1.0, 0.1)
    if st.session_state.decoding_mode == "top-k":
        st.session_state.decoding_params["k"] = st.sidebar.slider("k value", 1, 100, 5)
    elif st.session_state.decoding_mode == "diverse-beam-search":
        st.session_state.decoding_params["beam_width"] = st.sidebar.slider("beam width", 1, 10, 1)
        st.session_state.decoding_params["diversity_penalty"] = st.sidebar.slider("diversity penalty", 0.0, 1.0, 0.1)
    elif st.session_state.decoding_mode == "min-bayes-risk":
        st.session_state.decoding_params["num_candidates"] = st.sidebar.slider("Number of candidates", 1, 30, 4)
    
    if tab_selection == "Random Sample from Test Set":
        st.header("Generate from Test Dataset")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Number of samples in the test dataset
            st.write(f"Test dataset contains 5000 samples")
        
        with col2:
            # Button to generate a random sample
            if st.button("Generate Random Sample"):
                random_idx = np.random.randint(1, 5000)
                st.session_state.random_idx = random_idx
                st.session_state.ip, st.session_state.ip_mask, st.session_state.tg, st.session_state.tg_mask = get_sample(random_idx)

        # Display the selected sample
        if hasattr(st.session_state, 'random_idx'):
            st.subheader(f"Sample #{st.session_state.random_idx}")
            st.session_state.x = tokenizer.decode(st.session_state.ip.tolist()[0])
            st.session_state.y = tokenizer.decode(st.session_state.tg.tolist())
            # Display sample details in a table
            df = pd.DataFrame.from_dict({'X': [st.session_state.x], 'y': [st.session_state.y]})
            st.dataframe(df.T.reset_index(), width=800)
            
            # Generate output
            if st.button("Generate Sequence"):
                with st.spinner("Generating sequence..."):
                    print(f'Ip: {st.session_state.ip} | Mask: {st.session_state.ip_mask} \n Mode: {st.session_state.decoding_mode} | Params: {st.session_state.decoding_params}')
                    st.session_state.tok_output = genOp(
                        encoder, decoder, device,
                        st.session_state.ip,  # Convert to string for the placeholder function
                        st.session_state.ip_mask,
                        mode=st.session_state.decoding_mode,
                        **st.session_state.decoding_params
                    )
                    print(f'\n\n\nOutput: {st.session_state.tok_output} \n')
                    st.session_state.output = tokenizer.decode(st.session_state.tok_output)

            # Display output
            if hasattr(st.session_state, 'output'):
                st.subheader("Generated Sequence")
                st.write(st.session_state.output)

if __name__ == "__main__":
    main()
1