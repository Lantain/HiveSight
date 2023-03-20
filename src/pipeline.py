import os
from src.transformers.model import Transformer
from PIL import Image

class PipelineProcessor:
    src_dir: str = None
    images: list[str] = list()
    transformers: list[Transformer]

    def __init__(self, src_dir: str, transformers: list[Transformer]) -> None:
        self.transformers = transformers
        files = os.listdir(src_dir)
        for f in files:
            self.images.append(f"{src_dir}/{f}")

    def pipe_to_dir(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        for image in self.images:
            img = Image.open(image)
            
            for transformer in self.transformers:
                results = transformer.transform(img)
                img = results[0]
            
            img.save(f"{out_dir}/{os.path.basename(image)}")
    
    def get_image(self, n: int) -> Image.Image:
        if n < len(self.images) and n >= 0:
            img = Image.open(self.images[n])
            for tr in self.transformers:
                img = tr.transform(img)
            
            return img

class Pipeline:
    transformers: list[Transformer]

    def __init__(self, transformers: list[Transformer] = list()) -> None:
        self.transformers = transformers

    def from_dir(self, src_dir) -> PipelineProcessor:
        return PipelineProcessor(src_dir, self.transformers)
    
    def pipe_image(self, img_path: str) -> Image.Image:
        img = Image.open(img_path)
        for tr in self.transformers:
            img = tr.transform(img)
        
        return img
    
    def has_transformers(self): 
        return len(self.transformers) > 0