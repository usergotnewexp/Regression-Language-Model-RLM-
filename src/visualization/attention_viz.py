"""
Attention Visualization for Regression Language Model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


class AttentionVisualizer:
    """Visualize attention weights from RLM"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def get_attention_for_text(self, text: str) -> Tuple[List[str], np.ndarray, float]:
        """
        Get attention weights and prediction for a given text
        
        Returns:
            tokens: List of tokens
            attention: Attention weights [num_layers, num_heads, seq_len, seq_len]
            prediction: Model prediction
        """
        # Tokenize
        tokens = self.tokenizer.encode(text)
        tokens_list = self.tokenizer.decode_to_tokens(tokens)
        
        # Convert to tensor
        input_ids = torch.tensor([tokens])
        
        # Get prediction with attention
        with torch.no_grad():
            prediction, attention_weights = self.model(input_ids, return_attention=True)
        
        # Convert to numpy
        attention_np = [attn.cpu().numpy() for attn in attention_weights]
        attention_np = np.array(attention_np)  # [num_layers, batch, num_heads, seq_len, seq_len]
        attention_np = attention_np[:, 0, :, :, :]  # Remove batch dimension
        
        prediction_value = prediction.item()
        
        return tokens_list, attention_np, prediction_value
    
    def plot_attention_heatmap(
        self, 
        text: str, 
        layer: int = -1, 
        head: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot attention heatmap for a specific layer and head
        
        Args:
            text: Input text
            layer: Which transformer layer (-1 for last)
            head: Which attention head
            save_path: Path to save figure
            figsize: Figure size
        """
        tokens, attention, prediction = self.get_attention_for_text(text)
        
        # Select layer and head
        attn_matrix = attention[layer, head, :, :]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(
            attn_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )
        
        ax.set_xlabel('Key Tokens', fontsize=12)
        ax.set_ylabel('Query Tokens', fontsize=12)
        ax.set_title(
            f'Attention Heatmap - Layer {layer}, Head {head}\n'
            f'Prediction: {prediction:.2f}',
            fontsize=14,
            fontweight='bold'
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention heatmap to {save_path}")
        
        plt.show()
    
    def plot_attention_by_token(
        self,
        text: str,
        layer: int = -1,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Plot average attention weight received by each token
        (averaged across all heads and query positions)
        """
        tokens, attention, prediction = self.get_attention_for_text(text)
        
        # Average attention across heads and query positions
        attn_matrix = attention[layer]  # [num_heads, seq_len, seq_len]
        avg_attention = attn_matrix.mean(axis=(0, 1))  # Average over heads and queries
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(avg_attention / avg_attention.max())
        bars = ax.bar(range(len(tokens)), avg_attention, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Average Attention Weight', fontsize=12)
        ax.set_title(
            f'Token Importance (Layer {layer})\n'
            f'Text: "{text}"\n'
            f'Prediction: {prediction:.2f}',
            fontsize=14,
            fontweight='bold'
        )
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved token importance plot to {save_path}")
        
        plt.show()
        
        # Print top tokens
        top_indices = np.argsort(avg_attention)[-5:][::-1]
        print("\nTop 5 Most Attended Tokens:")
        for i, idx in enumerate(top_indices, 1):
            print(f"   {i}. '{tokens[idx]}' - Weight: {avg_attention[idx]:.4f}")
    
    def plot_multi_head_attention(
        self,
        text: str,
        layer: int = -1,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Plot attention patterns for all heads in a layer
        """
        tokens, attention, prediction = self.get_attention_for_text(text)
        
        num_heads = attention.shape[1]
        attn_matrix = attention[layer]  # [num_heads, seq_len, seq_len]
        
        # Calculate grid size
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for head in range(num_heads):
            sns.heatmap(
                attn_matrix[head],
                cmap='viridis',
                ax=axes[head],
                cbar=True,
                square=True,
                xticklabels=tokens if head >= num_heads - cols else [],
                yticklabels=tokens if head % cols == 0 else []
            )
            axes[head].set_title(f'Head {head}', fontsize=10)
            
            if head >= num_heads - cols:
                axes[head].set_xlabel('Keys', fontsize=8)
            if head % cols == 0:
                axes[head].set_ylabel('Queries', fontsize=8)
        
        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(
            f'Multi-Head Attention Patterns (Layer {layer})\n'
            f'Prediction: {prediction:.2f}',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-head attention plot to {save_path}")
        
        plt.show()
    
    def create_interactive_attention_plot(
        self,
        text: str,
        layer: int = -1,
        save_path: Optional[str] = None
    ):
        """
        Create interactive attention visualization using Plotly
        """
        tokens, attention, prediction = self.get_attention_for_text(text)
        
        num_heads = attention.shape[1]
        attn_matrix = attention[layer]  # [num_heads, seq_len, seq_len]
        
        # Create subplot for each head
        fig = go.Figure()
        
        # Add heatmap for first head (default)
        fig.add_trace(
            go.Heatmap(
                z=attn_matrix[0],
                x=tokens,
                y=tokens,
                colorscale='Viridis',
                showscale=True,
                hovertemplate='Query: %{y}<br>Key: %{x}<br>Attention: %{z:.4f}<extra></extra>'
            )
        )
        
        # Create buttons for each head
        buttons = []
        for head in range(num_heads):
            buttons.append(
                dict(
                    label=f'Head {head}',
                    method='update',
                    args=[
                        {'z': [attn_matrix[head]]},
                        {'title': f'Layer {layer}, Head {head} - Prediction: {prediction:.2f}'}
                    ]
                )
            )
        
        fig.update_layout(
            title=f'Interactive Attention Visualization<br>Layer {layer}, Head 0 - Prediction: {prediction:.2f}',
            xaxis_title='Key Tokens',
            yaxis_title='Query Tokens',
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction='down',
                    showactive=True,
                    x=1.15,
                    y=1.0
                )
            ],
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive attention plot to {save_path}")
        
        fig.show()
    
    def compare_attention_patterns(
        self,
        texts: List[str],
        layer: int = -1,
        head: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 5)
    ):
        """
        Compare attention patterns across multiple examples
        """
        n_texts = len(texts)
        fig, axes = plt.subplots(1, n_texts, figsize=figsize)
        
        if n_texts == 1:
            axes = [axes]
        
        for i, text in enumerate(texts):
            tokens, attention, prediction = self.get_attention_for_text(text)
            attn_matrix = attention[layer, head, :, :]
            
            sns.heatmap(
                attn_matrix,
                xticklabels=tokens,
                yticklabels=tokens if i == 0 else [],
                cmap='viridis',
                ax=axes[i],
                cbar=True,
                square=True
            )
            
            axes[i].set_title(f'Pred: {prediction:.2f}', fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Keys', fontsize=9)
            if i == 0:
                axes[i].set_ylabel('Queries', fontsize=9)
            
            # Rotate x labels
            axes[i].tick_params(axis='x', rotation=45, labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)
        
        fig.suptitle(
            f'Attention Comparison (Layer {layer}, Head {head})',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention comparison to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # This would be used with actual model and tokenizer
    print("AttentionVisualizer class defined successfully")
    print("\nExample usage:")
    print("""
    from models.rlm import EnhancedRegressionLanguageModel
    from data.preprocessor import Tokenizer
    
    # Load model and tokenizer
    model = EnhancedRegressionLanguageModel.load('checkpoints/best_model.pt')
    tokenizer = Tokenizer.load('checkpoints/tokenizer.pkl')
    
    # Create visualizer
    viz = AttentionVisualizer(model, tokenizer)
    
    # Visualize attention
    text = "This product is amazing and works perfectly!"
    viz.plot_attention_heatmap(text, layer=-1, head=0)
    viz.plot_attention_by_token(text)
    viz.plot_multi_head_attention(text)
    viz.create_interactive_attention_plot(text)
    """)
