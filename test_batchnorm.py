
import torch
import torch.nn as nn

def test_bn():
    bn = nn.BatchNorm1d(10)
    bn.train()
    
    # Random vector
    x = torch.randn(1, 10)
    
    # Duplicated batch
    batch = torch.cat([x, x], dim=0)
    
    # Forward
    out = bn(batch)
    
    print("Input:\n", batch)
    print("Output:\n", out)
    print("Output std:", out.std(dim=0))
    
    if torch.allclose(out, torch.zeros_like(out), atol=1e-3):
        print("FAIL: Output is all zeros!")
    else:
        print("SUCCESS: Output is not zeros.")

if __name__ == "__main__":
    test_bn()
