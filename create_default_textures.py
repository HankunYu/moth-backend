#!/usr/bin/env python3
"""
Create default textures for moth population control
"""
import cv2
import numpy as np
import os

def create_default_texture(color, filename, size=(512, 512)):
    """Create a simple colored texture"""
    # Create base color
    if color == "red":
        base_color = (50, 50, 200)  # BGR format
    elif color == "blue":
        base_color = (200, 50, 50)
    elif color == "green":
        base_color = (50, 200, 50)
    elif color == "yellow":
        base_color = (50, 200, 200)
    elif color == "purple":
        base_color = (200, 50, 200)
    else:
        base_color = (100, 100, 100)  # gray
    
    # Create texture with some pattern
    texture = np.full((size[1], size[0], 3), base_color, dtype=np.uint8)
    
    # Add some noise for variation
    noise = np.random.randint(-30, 30, (size[1], size[0], 3))
    texture = np.clip(texture.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Add some simple patterns
    cv2.circle(texture, (size[0]//2, size[1]//2), size[0]//4, (255, 255, 255), 2)
    cv2.rectangle(texture, (size[0]//4, size[1]//4), (3*size[0]//4, 3*size[1]//4), (0, 0, 0), 2)
    
    return texture

def main():
    """Create default textures"""
    output_dir = "./default_textures/"
    os.makedirs(output_dir, exist_ok=True)
    
    colors = ["red", "blue", "green", "yellow", "purple", "gray"]
    
    for color in colors:
        texture = create_default_texture(color, f"default_{color}.png")
        output_path = os.path.join(output_dir, f"default_{color}.png")
        cv2.imwrite(output_path, texture)
        print(f"Created: {output_path}")
    
    print(f"Created {len(colors)} default textures in {output_dir}")

if __name__ == "__main__":
    main()