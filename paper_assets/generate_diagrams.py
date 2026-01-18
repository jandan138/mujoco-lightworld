try:
    from graphviz import Digraph
except ImportError:
    print("Error: 'graphviz' library not found. Please install it via 'pip install graphviz' and ensure Graphviz executable is in your PATH.")
    exit(1)

def create_system_architecture():
    dot = Digraph(comment='System Architecture', format='png')
    dot.attr(rankdir='LR', dpi='300')

    with dot.subgraph(name='cluster_env') as c:
        c.attr(label='Environment', style='dashed')
        c.node('Obs', 'Observation (s_t)')
        c.node('Reward', 'Reward (r_t)')

    with dot.subgraph(name='cluster_agent') as c:
        c.attr(label='PPO Agent', style='filled', color='lightgrey')
        c.node('Policy', 'Policy Network')
        c.node('Buffer', 'Replay Buffer')
        c.node('Action', 'Action (a_t)')
        
    with dot.subgraph(name='cluster_wm') as c:
        c.attr(label='World Model (Auxiliary)', style='filled', color='lightblue')
        c.node('Encoder', 'Encoder')
        c.node('Dynamics', 'Dynamics Model')
        c.node('Latent', 'Latent State (z_t)')
        c.node('PredLatent', "Predicted z'_{t+1}")
        c.node('Loss', 'WM Loss')

    # Edges
    dot.edge('Obs', 'Policy')
    dot.edge('Policy', 'Action')
    dot.edge('Action', 'Environment')
    dot.edge('Obs', 'Buffer')
    dot.edge('Action', 'Buffer')
    
    dot.edge('Buffer', 'Encoder')
    dot.edge('Encoder', 'Latent')
    dot.edge('Latent', 'Dynamics')
    dot.edge('Action', 'Dynamics')
    dot.edge('Dynamics', 'PredLatent')
    
    dot.edge('Loss', 'Policy', label='Aux Gradient', style='dotted')

    output_path = 'paper_assets/fig1_system_arch'
    dot.render(output_path, view=False)
    print(f"Generated {output_path}.png")

def create_network_architecture():
    dot = Digraph(comment='Network Architecture', format='png')
    dot.attr(rankdir='TD', dpi='300')

    # Encoder
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder E_phi')
        c.node('In_S', 'Input: State s_t')
        c.node('Enc_FC1', 'FC + ReLU (64)')
        c.node('Enc_FC2', 'FC + ReLU (64)')
        c.node('Out_Z', 'Output: Latent z_t')
        
        c.edge('In_S', 'Enc_FC1')
        c.edge('Enc_FC1', 'Enc_FC2')
        c.edge('Enc_FC2', 'Out_Z')

    # Dynamics
    with dot.subgraph(name='cluster_dynamics') as c:
        c.attr(label='Dynamics D_psi')
        c.node('In_Z', 'Input: z_t')
        c.node('In_A', 'Input: a_t')
        c.node('Dyn_FC1', 'FC + ReLU (64)')
        c.node('Dyn_FC2', 'FC + ReLU (64)')
        c.node('Out_Z_next', 'Output: Predicted z_{t+1}')
        
        c.edge('In_Z', 'Dyn_FC1')
        c.edge('In_A', 'Dyn_FC1')
        c.edge('Dyn_FC1', 'Dyn_FC2')
        c.edge('Dyn_FC2', 'Out_Z_next')

    output_path = 'paper_assets/fig2_network_arch'
    dot.render(output_path, view=False)
    print(f"Generated {output_path}.png")

if __name__ == '__main__':
    create_system_architecture()
    create_network_architecture()
