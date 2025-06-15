
"""
Basic Stable Diffusion Image Generator
A simple script to generate images from text prompts using Hugging Face Diffusers
"""

import os
import torch
import argparse
from datetime import datetime
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        """
        Initialize the image generator
        
        Args:
            model_id: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        print("🚀 Initializing Stable Diffusion Image Generator...")
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"📱 Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 VRAM: {total_memory:.1f} GB")
        
        # Set up model cache directory
        cache_dir = Path.home() / ".cache" / "huggingface" / "diffusers"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the pipeline
        print(f"📦 Loading model: {model_id}")
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,  # Disable for faster loading
                requires_safety_checker=False,
                cache_dir=cache_dir
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable optimizations
            if self.device == "cuda":
                # Memory optimizations
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("✅ XFormers enabled for better performance")
                except ImportError:
                    print("⚠️ XFormers not available, using default attention")
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate(self, prompt, negative_prompt="", width=512, height=512, 
                 num_steps=20, guidance_scale=7.5, seed=None):
        """
        Generate an image from a text prompt
        
        Args:
            prompt: Text description of the image to generate
            negative_prompt: What to avoid in the image
            width: Image width (must be multiple of 8)
            height: Image height (must be multiple of 8)
            num_steps: Number of denoising steps (more = better quality, slower)
            guidance_scale: How closely to follow the prompt (7-15 typical)
            seed: Random seed for reproducibility
            
        Returns:
            PIL Image object
        """
        print(f"🎨 Generating image...")
        print(f"📝 Prompt: {prompt}")
        if negative_prompt:
            print(f"🚫 Negative: {negative_prompt}")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            print(f"🎲 Seed: {seed}")
        
        start_time = time.time()
        
        try:
            # Generate image
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator().manual_seed(seed) if seed else None
                )
            
            image = result.images[0]
            generation_time = time.time() - start_time
            print(f"⏱️ Generated in {generation_time:.2f} seconds")
            
            return image
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            raise
    
    def save_image(self, image, prompt, output_dir="./output"):
        """
        Save generated image with metadata
        
        Args:
            image: PIL Image to save
            prompt: Original prompt used
            output_dir: Directory to save images
            
        Returns:
            Path to saved image
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean prompt for filename (remove special characters)
        clean_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        clean_prompt = clean_prompt.replace(' ', '_')
        
        filename = f"{timestamp}_{clean_prompt}.png"
        filepath = output_path / filename
        
        # Save with metadata
        metadata = {
            "prompt": prompt,
            "timestamp": timestamp,
            "model": "stable-diffusion-v1-5"
        }
        
        # Add metadata to image
        from PIL.PngImagePlugin import PngInfo
        png_info = PngInfo()
        for key, value in metadata.items():
            png_info.add_text(key, str(value))
        
        image.save(filepath, pnginfo=png_info)
        print(f"💾 Saved: {filepath}")
        
        return filepath

def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
    parser.add_argument("prompt", help="Text prompt for image generation")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Model ID")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = ImageGenerator(model_id=args.model, device=args.device)
        
        # Generate image
        image = generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
        
        # Save image
        filepath = generator.save_image(image, args.prompt, args.output)
        
        print(f"🎉 Success! Image saved to: {filepath}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Generation cancelled by user")
    except Exception as e:
        print(f"💥 Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
