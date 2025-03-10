import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from src.models.resnet.resnet_cifar import resnet20_cifar10

class Hook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()

class DeepDream:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        layer = dict([*self.model.named_modules()])[target_layer]
        self.hook = Hook(layer)
        self.mean = [0.4914, 0.4822, 0.4465]  # CIFAR-10 mean
        self.std = [0.2023, 0.1994, 0.2010]

    def preprocess(self, image, size):
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        tensor = transform(image).unsqueeze(0).requires_grad_(True)
        return tensor

    def deprocess(self, tensor):
        tensor = tensor.detach().cpu().squeeze()
        for c in range(3):
            tensor[c] = tensor[c] * self.std[c] + self.mean[c]
        tensor = torch.clamp(tensor, 0, 1)
        img = transforms.ToPILImage()(tensor)
        return img

    def dream(self, input_tensor, iterations, step_size):
        for i in range(iterations):
            self.model.zero_grad()
            input_tensor = input_tensor.requires_grad_(True)
            input_tensor.retain_grad()
            self.model(input_tensor)
            # Loss: norm of the activations at the hooked layer.
            loss = self.hook.features.norm()
            loss.backward(retain_graph=True)

            grad = input_tensor.grad
            # Normalize the gradient to stabilize updates.
            input_tensor.data += step_size * grad / (grad.std() + 1e-8)
            input_tensor.grad.data.zero_()
        return input_tensor.detach()

    def dream_octaves(self, image, iterations=20, octaves=3, octave_scale=1.4, step_size=0.02):
        # image = np.array(image)
        octaves_images = [image]

        # Generate octave scales
        for _ in range(octaves - 1):
            hw = [int(dim / octave_scale) for dim in octaves_images[-1].shape[:2]]
            octave_image = Image.fromarray(octaves_images[-1]).resize(hw[::-1], Image.LANCZOS)
            octaves_images.append(np.array(octave_image))

        detail = torch.zeros_like(self.preprocess(Image.fromarray(octaves_images[-1]), octaves_images[-1].shape[0]))

        for octave_idx in reversed(range(octaves)):
            input_img = Image.fromarray(octaves_images[octave_idx])
            input_tensor = self.preprocess(input_img, input_img.size[0]) + detail
            input_tensor = self.dream(input_tensor, iterations, step_size)

            if octave_idx > 0:
                upscaled = nn.functional.interpolate(input_tensor, size=octaves_images[octave_idx - 1].shape[:2], mode='bilinear', align_corners=False)
                detail = upscaled - self.preprocess(Image.fromarray(octaves_images[octave_idx - 1]), octaves_images[octave_idx - 1].shape[0])

        return self.deprocess(input_tensor)

    def remove_hook(self):
        self.hook.hook.remove()

if __name__ == "__main__":
    import torch
    # from torchvision.models import resnet18  # Replace with your CIFAR-trained model if available

    # model = resnet18(pretrained=True).eval()
    model = resnet20_cifar10(pretrained=True).eval()

    target_layer = 'layer3.2.conv2'
    dreamer = DeepDream(model, target_layer)

    # image_path = "your_image.jpg"
    # original_image = Image.open(image_path).convert('RGB')
    original_image = (np.random.rand(32, 32, 3)*255).round().astype(np.uint8)

    dreamed_image = dreamer.dream_octaves(
        image=original_image,
        iterations=10,
        octaves=3,
        octave_scale=1.4,
        step_size=0.02
    )

    # dreamed_img = dreamer.deprocess(dreamed_image)
    dreamed_image.save("deepdream_result.jpg")
    dreamed_image.show()

    dreamer.remove_hook()
