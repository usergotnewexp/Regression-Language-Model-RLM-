"""
Interactive Streamlit Demo for Regression Language Model
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.rlm import EnhancedRegressionLanguageModel, create_rlm
from src.data.preprocessor import SimpleTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


# Page config
st.set_page_config(
    page_title="Regression Language Model Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer (cached)"""
    try:
        # Try to load saved tokenizer and model from checkpoints
        ckpt_dir = Path("checkpoints")
        tokenizer_path = ckpt_dir / "tokenizer.pkl"
        model_path = ckpt_dir / "best_model.pt"
        meta_path = ckpt_dir / "model_meta.json"

        if tokenizer_path.exists() and model_path.exists():
            import joblib, json
            tokenizer = joblib.load(tokenizer_path)

            # Determine model size from metadata (if available)
            model_size = 'base'
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        model_size = meta.get('model_size', 'base')
                except Exception:
                    pass

            # Build model and load state dict
            model = create_rlm(vocab_size=tokenizer.vocab_size, model_size=model_size)
            ckpt = torch.load(model_path, map_location='cpu')
            # Trainer checkpoints store a dict with 'model_state_dict'
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                model.load_state_dict(ckpt)

            model.eval()
            return model, tokenizer

        # Fallback: create a demo tokenizer and untrained model
        tokenizer = SimpleTokenizer()
        # Build vocabulary from demo examples
        demo_texts = [
            "This is amazing!",
            "Not great",
            "Pretty good",
            "Terrible experience",
            "Absolutely loved it"
        ]
        tokenizer.fit(demo_texts)

        model = create_rlm(vocab_size=tokenizer.vocab_size, model_size='base')
        model.eval()

        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def predict_with_model(model, tokenizer, text):
    """Make prediction with the model"""
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        # Regular prediction
        output = model(input_ids)
        prediction = output.item()
        
        # Prediction with uncertainty
        uncertainty = model.predict_with_uncertainty(input_ids, n_samples=30)
        
        # Get attention weights
        _, attention = model(input_ids, return_attention=True)
    
    return {
        'prediction': prediction,
        'uncertainty': uncertainty,
        'attention': attention,
        'tokens': tokenizer.decode_to_tokens(tokens)
    }


def plot_uncertainty(result):
    """Plot uncertainty visualization"""
    mean = result['uncertainty']['mean'][0, 0]
    std = result['uncertainty']['std'][0, 0]
    lower = result['uncertainty']['lower_95'][0, 0]
    upper = result['uncertainty']['upper_95'][0, 0]
    
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[lower, lower],
        mode='lines',
        line=dict(color='rgba(0,100,200,0.3)', width=2),
        name='Lower 95% CI',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[upper, upper],
        mode='lines',
        line=dict(color='rgba(0,100,200,0.3)', width=2),
        fill='tonexty',
        name='Upper 95% CI',
        showlegend=True
    ))
    
    # Add mean prediction
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[mean, mean],
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=10),
        name=f'Prediction: {mean:.2f}',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Prediction with 95% Confidence Interval",
        xaxis_title="",
        yaxis_title="Predicted Value",
        height=300,
        showlegend=True,
        xaxis=dict(showticklabels=False)
    )
    
    return fig


def plot_token_importance(tokens, attention):
    """Plot token importance from attention"""
    # Average attention across all layers, heads, and query positions
    avg_attention = []
    for layer_attn in attention:
        layer_avg = layer_attn[0].mean(axis=(0, 1)).cpu().numpy()
        avg_attention.append(layer_avg)
    
    avg_attention = np.array(avg_attention).mean(axis=0)
    
    # Create bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=tokens,
            y=avg_attention,
            marker=dict(
                color=avg_attention,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            )
        )
    ])
    
    fig.update_layout(
        title="Token Importance (Attention Weights)",
        xaxis_title="Tokens",
        yaxis_title="Average Attention",
        height=400
    )
    
    return fig


def main():
    # Header
    st.markdown('<div class="main-header">🎯 Regression Language Model Demo</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This demo showcases a **Transformer-based Regression Language Model** that predicts 
    continuous numerical values from text input. The model uses multi-head self-attention 
    to understand semantic relationships and provides uncertainty estimates for its predictions.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        domain = st.selectbox(
            "Domain",
            ["Restaurant Reviews", "Financial News", "Real Estate", "Custom"]
        )
        
        if domain == "Restaurant Reviews":
            st.info("📊 Predicting star ratings (1-5)")
            example_texts = [
                "The food was absolutely amazing!",
                "Service was slow and food was cold",
                "Pretty decent meal overall",
                "Best restaurant in town!",
                "Not worth the price"
            ]
        elif domain == "Financial News":
            st.info("📈 Predicting % price change")
            example_texts = [
                "Company reports record quarterly earnings",
                "CEO steps down amid controversy",
                "Strong growth forecast announced",
                "Product recall affects thousands",
                "Analyst upgrades stock rating"
            ]
        elif domain == "Real Estate":
            st.info("🏠 Predicting property price")
            example_texts = [
                "Spacious 3BR apartment in downtown",
                "Cozy 2BR house with large backyard",
                "Luxurious 5BR mansion with pool",
                "Studio apartment near metro",
                "Modern 4BR townhouse"
            ]
        else:
            example_texts = []
        
        st.markdown("---")
        st.header("📚 About")
        st.markdown("""
        **Key Features:**
        - 🎯 Continuous value prediction
        - 🔍 Attention visualization
        - 📊 Uncertainty estimation
        - 🚀 Real-time inference
        
        **Model Architecture:**
        - Transformer Encoder (4 layers)
        - Multi-head Attention (8 heads)
        - 256-dimensional embeddings
        """)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load model. Please check the error message above.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Input Text")
        
        # Text input (persist in session state so example buttons work)
        if 'user_text' not in st.session_state:
            st.session_state['user_text'] = "This product is amazing and works perfectly!"

        user_text = st.text_area(
            "Enter text for prediction:",
            value=st.session_state['user_text'],
            height=100,
            key='user_text',
            help="Enter any text to get a numerical prediction"
        )
        
        # Example buttons
        if example_texts:
            st.markdown("**Or try these examples:**")
            cols = st.columns(len(example_texts))
            for i, (col, example) in enumerate(zip(cols, example_texts)):
                if col.button(f"Example {i+1}", key=f"example_{i}"):
                    # Set the session state and rerun so the text_area updates
                    st.session_state['user_text'] = example
                    st.experimental_rerun()
        
        predict_button = st.button("🎯 Predict", type="primary", use_container_width=True)
    
    with col2:
        st.header("⚡ Quick Stats")
        st.metric("Model Size", "424K params")
        st.metric("Inference Time", "~10ms")
        st.metric("Vocabulary", f"{tokenizer.vocab_size if tokenizer else 0} tokens")
    
    # Prediction
    if predict_button and user_text:
        with st.spinner("🔮 Making prediction..."):
            try:
                result = predict_with_model(model, tokenizer, user_text)
                
                # Display results
                st.markdown("---")
                st.header("📊 Results")
                
                # Prediction metrics
                col1, col2, col3 = st.columns(3)
                
                mean = result['uncertainty']['mean'][0, 0]
                std = result['uncertainty']['std'][0, 0]

                # Format outputs according to domain
                display_mean = mean
                display_text = None
                if domain == "Restaurant Reviews":
                    display_mean = float(np.clip(mean, 1.0, 5.0))
                    display_text = f"{display_mean:.2f} / 5.0"
                elif domain == "Financial News":
                    display_mean = float(mean)
                    display_text = f"{display_mean:.2f}%"
                elif domain == "Real Estate":
                    # Assume model predicts price in raw numbers
                    display_mean = float(mean)
                    display_text = f"${display_mean:,.0f}"
                else:
                    display_text = f"{display_mean:.2f}"
                
                with col1:
                    st.metric(
                        "Prediction", 
                        display_text,
                        help="Mean predicted value"
                    )
                
                with col2:
                    st.metric(
                        "Uncertainty (σ)", 
                        f"{std:.2f}",
                        help="Standard deviation of predictions"
                    )
                
                with col3:
                    confidence = max(0, min(100, 100 * (1 - std / abs(mean)) if mean != 0 else 0))
                    st.metric(
                        "Confidence", 
                        f"{confidence:.1f}%",
                        help="Model confidence in prediction"
                    )
                
                # Uncertainty visualization
                st.plotly_chart(plot_uncertainty(result), use_container_width=True)
                
                # Token importance
                st.subheader("🔍 Token Importance Analysis")
                st.plotly_chart(
                    plot_token_importance(result['tokens'], result['attention']),
                    use_container_width=True
                )
                
                # Detailed breakdown
                with st.expander("📝 Detailed Breakdown"):
                    st.markdown("**Tokenized Input:**")
                    st.code(' | '.join(result['tokens']))
                    
                    st.markdown("**Confidence Interval:**")
                    st.write(f"- Lower bound (95%): {result['uncertainty']['lower_95'][0, 0]:.2f}")
                    st.write(f"- Mean prediction: {mean:.2f}")
                    st.write(f"- Upper bound (95%): {result['uncertainty']['upper_95'][0, 0]:.2f}")
                    
                    st.markdown("**Model Interpretation:**")
                    if std < 0.5:
                        st.success("✅ High confidence prediction - model is very certain")
                    elif std < 1.0:
                        st.warning("⚠️ Moderate confidence - some uncertainty in prediction")
                    else:
                        st.error("❌ Low confidence - high uncertainty, consider more context")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using PyTorch, Transformers, and Streamlit<br>
        <a href='https://github.com/yourusername/regression-language-model'>
            View on GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
