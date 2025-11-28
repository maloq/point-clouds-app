
import torch
import numpy as np
import sys
import traceback
import time

# Mock imports
sys.path.append('.')
from structures.generators import generate_point_cloud, StructureType, normalize_point_cloud
from structures.deformations import Deformation
from networks.training import AlignmentNetwork, Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def test_training():
    try:
        struct_type = "lattice_bcc"
        n_points = 64
        rot_x, rot_y, rot_z = 45, 45, 0
        noise = 0.0
        lr = 0.001
        epochs = 2
        encoders = ["mlp", "pointnet", "gnn"]
        heads = ["rotation", "flow", "combined"]
        
        for enc in encoders:
            for head in heads:
                print(f"\nTesting {enc} + {head}...")
                
                print("Generating data...")
                source = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
                source = normalize_point_cloud(source)
                
                deform = Deformation(device=DEVICE)
                deform.rotate(euler_angles=(np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)))
                target = deform(source)
                
                print("Initializing network...")
                model = AlignmentNetwork(encoder_type=enc, head_type=head, num_points=int(n_points))
                trainer = Trainer(model, lr=lr, device=DEVICE)
                
                # Create batch of size 2 with noisy copy
                source_noisy = source + torch.randn_like(source) * 0.01
                target_noisy = target + torch.randn_like(target) * 0.01
                
                source_batch = torch.stack([source, source_noisy])
                target_batch = torch.stack([target, target_noisy])
                
                print("Starting training loop...")
                for i in range(int(epochs)):
                    loss = trainer.train_step(source_batch, target_batch)
                    # print(f"Epoch {i}, Loss: {loss}")
                    
                    with torch.no_grad():
                        outputs = model(source_batch, target_batch)
                        
                print(f"Success: {enc} + {head}")
        
    except Exception as e:
        print(f"Caught exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_training()
