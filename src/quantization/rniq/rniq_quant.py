import lightning.pytorch as pl
import torch.nn.functional as F
import torch

from src.models.resnet.resnet_cifar import resnet20_cifar10_new, resnet20_cifar10
from src.quantization.abc.abc_quant import BaseQuant
from src.quantization.rniq.layers.rniq_conv2d import NoisyConv2d
from src.quantization.rniq.layers.rniq_linear import NoisyLinear
from src.quantization.rniq.layers.rniq_act import NoisyAct
from src.quantization.rniq.utils.model_helper import ModelHelper
from src.quantization.rniq.rniq_loss import PotentialLoss
from src.quantization.rniq.utils import model_stats, hooks
from src.aux.qutils import attrsetter, is_biased
from src.aux.loss.hellinger import HellingerLoss
from src.aux.loss.symm_ce_loss import SymmetricalCrossEntropyLoss
from src.aux.loss.distill_ce import CrossEntropyLoss
from src.aux.loss.symm_kl_loss import SymmetricalKL
from src.aux.loss.kl_loss import KL
from src.aux.loss.jsdloss import JSDLoss

from torch import nn
from copy import deepcopy
from operator import attrgetter
from collections import OrderedDict

from torchvision import transforms
 
def deprocess_image(image_tensor):
    """Convert a tensor back to a displayable image."""
    image_tensor = image_tensor.squeeze(0).detach().cpu()
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
    image_tensor = image_tensor * std + mean
    image_tensor = torch.clamp(image_tensor, 0, 1)
    return transforms.ToPILImage()(image_tensor)


class RNIQQuant(BaseQuant):
    def module_mappings(self):
        return {
            nn.Conv2d: NoisyConv2d,
            nn.Linear: NoisyLinear,
        }

    def get_distill_loss(self, qmodel):
        if self.config.quantization.distillation:
            config_loss = self.config.quantization.distillation_loss
            if config_loss == "Cross-Entropy":
                return CrossEntropyLoss()
            elif config_loss == "Symmetrical Cross-Entropy":
                return SymmetricalCrossEntropyLoss()
            elif config_loss == "L1":
                return torch.nn.L1Loss()
            elif config_loss == "L2":
                return torch.nn.MSELoss()
            elif config_loss == "KL":
                return KL()
            elif config_loss == "Hellinger":
                return HellingerLoss()
            elif config_loss == "Symmetrical KL":
                return SymmetricalKL()
            elif config_loss == "JSD":
                return JSDLoss()
            else:
                raise NotImplementedError("Loss type are invalid! \
                                          Valid options are: \
                                            [Cross-Entropy,Symmetrical Cross-Entropy, L1, L2, KL, Hellinger]")
        else:
            return qmodel.criterion

    def quantize(self, lmodel: pl.LightningModule, in_place=False):
        if self.config.quantization.distillation:
            if not self.config.quantization.distillation_teacher:
                tmodel = deepcopy(lmodel).eval()
            else: #XXX fix me
                tmodel = resnet20_cifar10_new(pretrained=True)
        if in_place:
            qmodel = lmodel
        else:
            qmodel = deepcopy(lmodel)

        # qmodel.model = resnet20_cifar10(pretrained=False)

        layer_names, layer_types = zip(
            *[(n, type(m)) for n, m in qmodel.model.named_modules()]
        )

        # The part where original LModule structure gets changed
        qmodel._noise_ratio = torch.tensor(1.0)
        qmodel.qscheme = self.qscheme

        if self.config.quantization.distillation:
            qmodel.tmodel = tmodel.requires_grad_(False)
            
            # chosen layer to propagate back from
            # chosen_module = tmodel.model.features.stage3.unit3.body.conv2.conv
            # chosen_module = tmodel.model.layer3[2].conv2 # for cifar10 old
            chosen_module = tmodel.model.features.stage3.unit3.body.conv2.conv # for cifar100
            # chosen_module = tmodel.model.features.stage2.unit3.body.conv2.conv
            ###
            qmodel.tmodel.hook = hooks.ActivationHook(chosen_module)

            # for module in tmodel.model.modules():
            #     if isinstance(module, nn.Conv2d):
            #         module.hook = hooks.ActivationHook(module)


        qmodel.wrapped_criterion = PotentialLoss(
            criterion=self.get_distill_loss(qmodel=qmodel),
            p=1,
            a=self.act_bit,
            w=self.weight_bit,
        )

        qmodel.noise_ratio = RNIQQuant.noise_ratio.__get__(
            qmodel, type(qmodel))

        # Important step. Replacing training and validation steps
        # with alternated ones.
        if self.config.quantization.distillation:
            qmodel.training_step = RNIQQuant.distillation_deepdream_noisy_training_step.__get__(
                qmodel, type(qmodel)
            )

            # qmodel.training_step = RNIQQuant.distillation_noisy_training_step.__get__(
                # qmodel, type(qmodel)
            # )
        else:
            qmodel.training_step = RNIQQuant.noisy_training_step.__get__(
                qmodel, type(qmodel)
            )

        qmodel.validation_step = RNIQQuant.noisy_validation_step.__get__(
            qmodel, type(qmodel)
        )
        qmodel.test_step = RNIQQuant.noisy_test_step.__get__(
            qmodel, type(qmodel))

        # Replacing layers directly
        qlayers = self._get_layers(
            lmodel.model, exclude_layers=self.excluded_layers)
        for layer in qlayers.keys():
            module = attrgetter(layer)(lmodel.model)
            if module.kernel_size != (1,1):
                print(layer + " " + repr(module.kernel_size))
                preceding_layer_type = layer_types[layer_names.index(layer) - 1]
                if issubclass(preceding_layer_type, nn.ReLU): #XXX: hack shoul be changed through config
                    qmodule = self._quantize_module(
                        module, signed_Activations=False)
                else:
                    qmodule = self._quantize_module(
                        module, signed_Activations=False)

                attrsetter(layer)(qmodel.model, qmodule)

        if self.config.quantization.freeze_batchnorm:
            RNIQQuant.freeze_all_batchnorm_layers(qmodel)                

        return qmodel
    
    def freeze_all_batchnorm_layers(model, freeze=True):
        # Freezes all batch normalization layers in the model. 
        # This means they won't update running means/variances 
        # during training and their parameters won't receive gradients.
        for module in model.modules():
            # Check for any batch norm variant
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Switch to evaluation mode (affects running stats)
                module.eval()
                # Freeze BN params
                module.weight.requires_grad = not freeze
                module.bias.requires_grad = not freeze

    @staticmethod
    def noise_ratio(self, x=None):
        if x != None:
            for module in self.modules():
                if hasattr(module, "_noise_ratio"):
                    module._noise_ratio.data = x.clone().detach()
        return self._noise_ratio

    @staticmethod  # yes, it's a static method with self argument
    def noisy_step(self, x):
        # now that we set qmodule.qscheme, we can address it in replaced step
        return (self.model(x), *ModelHelper.get_model_values(self.model, self.qscheme))

    @staticmethod
    def distillation_noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = RNIQQuant.noisy_step(self, inputs)

        self.tmodel.eval()
        fp_outputs = self.tmodel(inputs)
        loss = self.wrapped_criterion(outputs, fp_outputs)

        self.log("Loss/FP loss", F.cross_entropy(fp_outputs, targets))
        self.log("Loss/Train loss", loss, prog_bar=True)
        self.log(
            "Loss/Base train loss", self.wrapped_criterion.base_loss, prog_bar=True
        )
        self.log("Loss/Wloss", self.wrapped_criterion.wloss, prog_bar=False)
        self.log("Loss/Aloss", self.wrapped_criterion.aloss, prog_bar=False)
        self.log(
            "Loss/Weight reg loss",
            self.wrapped_criterion.weight_reg_loss,
            prog_bar=False,
        )
        self.log("LR", self.lr, prog_bar=True)

        return loss

    @staticmethod
    def noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = RNIQQuant.noisy_step(self, inputs)
        loss = self.wrapped_criterion(outputs, targets)

        self.log("Loss/Train loss", loss, prog_bar=True)
        self.log(
            "Loss/Base train loss", self.wrapped_criterion.base_loss, prog_bar=True
        )
        self.log("Loss/Wloss", self.wrapped_criterion.wloss, prog_bar=False)
        self.log("Loss/Aloss", self.wrapped_criterion.aloss, prog_bar=False)
        self.log(
            "Loss/Weight reg loss",
            self.wrapped_criterion.weight_reg_loss,
            prog_bar=False,
        )
        self.log("LR", self.lr, prog_bar=True)

        return loss
    
    @staticmethod
    def distillation_deepdream_noisy_training_step(self, batch, batch_idx):
        inputs, targets = batch
        # mean = torch.Tensor([0.4914, 0.4822, 0.4465]).to(inputs.device)
        # std = torch.Tensor([0.247, 0.243, 0.261]).to(inputs.device)

        # noise = torch.randn_like(inputs) * std[None, :, None, None] + mean[None, :, None, None]
        noise = torch.randn_like(inputs)
        noise.requires_grad_(True)

        # fp_outputs = self.tmodel(inputs)

        # Kinda deepdream of some sort
        ##############################
        self.tmodel.eval()

        loss_ = torch.zeros((1), device=inputs.device)

        # inputs.requires_grad_(True)
        fp_outputs = self.tmodel(inputs)
        ref_fmap = self.tmodel.hook.feature_map.clone().detach()

        noise_fp_outputs = self.tmodel(noise)
        init_noise_fmap = self.tmodel.hook.feature_map

        init_loss = F.mse_loss(init_noise_fmap, ref_fmap)

        # loss_ = self.tmodel.hook.feature_map.norm()
        init_loss.backward(retain_graph=True)

        step_size = 0.5
        with torch.no_grad():
            # inputs = inputs.detach() + step_size * inputs.grad / (inputs.grad.std() + 1e-8)
            # noise += step_size * inputs.grad.data / (inputs.grad.std() + 1e-8)
            noise.data -= step_size * noise.grad.data.detach() / (noise.grad.std() + 1e-8)
            # inputs.data = inputs.data + step_size * inputs.grad.data / (inputs.grad.std() + 1e-8)
            noise.grad.data.zero_()

        for i in range(10):
            # print(f"{i} ACC = {((self.tmodel(inputs).argmax(axis=1) == self.tmodel(noise).argmax(axis=1)).sum() / noise.shape[0]).item()}")

            # for module in self.tmodel.model.modules():
                # if isinstance(module, nn.Conv2d):
                    # loss_ += module.hook.feature_map.norm()
            
            noise.requires_grad_(True)

            fp_outputs = self.tmodel(noise)
            noise_fmap = self.tmodel.hook.feature_map

            loss_ = F.mse_loss(noise_fmap, ref_fmap)
            # loss_ = F.l1_loss(noise_fmap, ref_fmap)
            # loss_ = self.tmodel.hook.feature_map.norm()
            loss_.backward(retain_graph=True)

            self.log("Loss/Dream loss", loss_, prog_bar=True)

            # print(f"Loss = {loss_}")

            step_size = 0.05
            with torch.no_grad():
                # inputs = inputs.detach() + step_size * inputs.grad / (inputs.grad.std() + 1e-8)
                noise.data = noise.data - step_size * noise.grad.data / (noise.grad.std() + 1e-8)
                # noise.data = noise.data - step_size * noise.grad.data
                # inputs.data = inputs.data + step_size * inputs.grad.data / (inputs.grad.std() + 1e-8)
                # noise.data = noise.data.div(noise.std())
                noise.grad.data.zero_()
            
        ##############################

        # noise.data = noise.data / noise.data.std()   

        # print("ACC = ", ((self.tmodel(inputs).argmax(axis=1) == self.tmodel(noise).argmax(axis=1)).sum() / noise.shape[0]).item())
        # exit(0)
        inputs.requires_grad_(False)
        # fp_outputs_ = self.tmodel(inputs)
        fp_outputs_ = self.tmodel(noise)
        # outputs = RNIQQuant.noisy_step(self, inputs)
        outputs = RNIQQuant.noisy_step(self, noise)
        # outputs = RNIQQuant.noisy_step(self, inputs)
        
        loss = self.wrapped_criterion(outputs, fp_outputs_)

        self.log("Loss/FP loss", F.cross_entropy(fp_outputs_, targets))
        self.log("Loss/Train loss", loss, prog_bar=True)
        self.log(
            "Loss/Base train loss", self.wrapped_criterion.base_loss, prog_bar=True
        )
        self.log("Loss/Wloss", self.wrapped_criterion.wloss, prog_bar=False)
        self.log("Loss/Aloss", self.wrapped_criterion.aloss, prog_bar=False)
        self.log(
            "Loss/Weight reg loss",
            self.wrapped_criterion.weight_reg_loss,
            prog_bar=False,
        )
        self.log("LR", self.lr, prog_bar=True)

        return loss
        
        

    @staticmethod
    def noisy_validation_step(self, val_batch, val_index):
        inputs, targets = val_batch

        # targets = self.tmodel(inputs)
        # self.noise_ratio(0.0)
        outputs = RNIQQuant.noisy_step(self, inputs)        

        val_loss = self.criterion(outputs[0], targets)
        for name, metric in self.metrics:
            metric_value = metric(outputs[0], targets)
            # metric_value = metric(outputs, targets)
            self.log(f"Metric/{name}", metric_value, prog_bar=False)
            self.log(f"Metric/ns_{name}", metric_value * model_stats.is_converged(self), prog_bar=False) 

        # Not very optimal approach. Cycling through model two times..
        self.log(
            "Mean weights bit width",
            model_stats.get_weights_bit_width_mean(self.model),
            prog_bar=False,
        )
        self.log(
            "Actual weights bit width",
            model_stats.get_true_weights_width(self.model, max=False),
            prog_bar=False
        )
        self.log(
            "Actual weights max bit width",
            model_stats.get_true_weights_width(self.model),
            prog_bar=False
        )
        self.log(
            "Mean activations bit width",
            model_stats.get_activations_bit_width_mean(self.model),
            prog_bar=False,
        )
        self.log(
            "Actual activations bit widths",
            model_stats.get_true_activations_width(self.model, max=False),
            prog_bar=False
        )
        self.log(
            "Actual activations max bit widths",
            model_stats.get_true_activations_width(self.model),
            prog_bar=False
        )

        self.log("Loss/Validation loss", val_loss, prog_bar=False)


    @staticmethod
    def noisy_test_step(self, test_batch, test_index):
        inputs, targets = test_batch
        # self.noise_ratio(0.0)
        outputs = RNIQQuant.noisy_step(self, inputs)

        test_loss = self.criterion(outputs[0], targets)
        for name, metric in self.metrics:
            metric_value = metric(outputs[0], targets)
            self.log(f"{name}", metric_value, prog_bar=False)

        self.log("test_loss", test_loss, prog_bar=True)

    def _init_config(self):
        if self.config:
            self.quant_config = self.config.quantization
            self.act_bit = self.quant_config.act_bit
            self.weight_bit = self.quant_config.weight_bit
            self.excluded_layers = self.quant_config.excluded_layers
            self.qscheme = self.quant_config.qscheme

    def _quantize_module(self, module, signed_Activations):
        if isinstance(module, nn.Conv2d):
            qmodule = self._quantize_module_conv2d(module)
        elif isinstance(module, nn.Linear):
            qmodule = self._quantize_module_linear(module)
        else:
            raise NotImplementedError(f"Module not supported {type(module)}")

        qmodule.weight = module.weight

        if is_biased(module):
            qmodule.bias = module.bias

        # qmodule = self._get_quantization_sequence(qmodule, signed_Activations, channels=qmodule.in_channels)
        qmodule = self._get_quantization_sequence(qmodule, signed_Activations)

        return qmodule

    def _get_quantization_sequence(self, qmodule, signed_activations):
        disabled = False
        if self.config.quantization.act_bit == -1 or self.config.quantization.act_bit > 20:
            disabled = True
        sequence = nn.Sequential(
            OrderedDict(
                [
                    ("activations_quantizer", NoisyAct(signed=signed_activations, disable=disabled)),
                    ("0", qmodule),
                ]
            )
        )

        return sequence

    def _quantize_module_conv2d(self, module: nn.Conv2d):
        return NoisyConv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            is_biased(module),
            module.padding_mode,
            qscheme=self.qscheme,
            log_s_init=-12,
        )

    def _quantize_module_linear(self, module: nn.Linear):
        return NoisyLinear(
            module.in_features,
            module.out_features,
            is_biased(module),
            qscheme=self.qscheme,
            log_s_init=-12,
        )
