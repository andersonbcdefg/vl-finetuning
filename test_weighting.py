#!/usr/bin/env python3

"""Test script for dataset weighting functionality."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vl_utils.data import load_data

def test_basic_functionality():
    """Test that the basic functionality still works without weights."""
    print("=== Testing basic functionality (no weights) ===")
    try:
        train_ds, test_ds = load_data(
            datasets=["webclick"], 
            test_size=10,
            seed=42
        )
        print(f"✓ Basic loading works: train={len(train_ds)}, test={len(test_ds)}")
        return True
    except Exception as e:
        print(f"✗ Basic loading failed: {e}")
        return False

def test_weighted_functionality():
    """Test the weighted functionality with multiple datasets."""
    print("\n=== Testing weighted functionality ===")
    try:
        # Test with equal weights
        train_ds, test_ds = load_data(
            datasets=["webclick", "groundui-1k"],
            weights={"webclick": 1.0, "groundui-1k": 1.0},
            total_samples=200,  # Small number for testing
            test_size=10,
            seed=42
        )
        print(f"✓ Weighted loading works: train={len(train_ds)}, test={len(test_ds)}")
        
        # Test with different weights
        train_ds2, test_ds2 = load_data(
            datasets=["webclick", "groundui-1k"],
            weights={"webclick": 3.0, "groundui-1k": 1.0},
            total_samples=400,
            test_size=10,
            seed=42
        )
        print(f"✓ Different weights work: train={len(train_ds2)}, test={len(test_ds2)}")
        return True
    except Exception as e:
        print(f"✗ Weighted loading failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing dataset weighting implementation...")
    
    basic_ok = test_basic_functionality()
    weighted_ok = test_weighted_functionality()
    
    if basic_ok and weighted_ok:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)