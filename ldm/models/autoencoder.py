import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from packaging import version
from ldm.modules.ema import LitEma
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import *

from ldm.util import instantiate_from_config



class VQFlowNet(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim # 3
        self.n_embed = n_embed # 8192
        self.image_key = image_key # 'image'
        self.encoder = FlowEncoder(**ddconfig)
        self.decoder = FlowDecoderWithResidual(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor
        self.h0 = None
        self.w0 = None
        self.h_padded = None
        self.w_padded = None
        self.pad_h = 0
        self.pad_w = 0

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x, ret_feature=False):
        '''
        Set ret_feature = True when encoding conditions in ddpm
        '''
        # Pad the input first so its size is deividable by 8.
        # this is to tolerate different f values, various size inputs, 
        # and some operations in the DDPM unet model.
        self.h0, self.w0 = x.shape[2:]
        # 8: window size for max vit
        # 2**(nr-1): f 
        # 4: factor of downsampling in DDPM unet
        min_side = 8 * 2**(self.encoder.num_resolutions-1) * 4
        min_side = min_side // 2 if self.h0 <= 256 else min_side
        if self.h0 % min_side != 0:
            pad_h = min_side - (self.h0 % min_side)
            if pad_h == self.h0: # this is to avoid padding 256 patches
                pad_h = 0
            x = F.pad(x, (0, 0, 0, pad_h), mode='reflect')
            self.h_padded = True
            self.pad_h = pad_h

        if self.w0 % min_side != 0:
            pad_w = min_side - (self.w0 % min_side)
            if pad_w == self.w0:
                pad_w = 0
            x = F.pad(x, (0, pad_w, 0, 0), mode='reflect')
            self.w_padded = True
            self.pad_w = pad_w

        phi_list = None
        if ret_feature:
            h, phi_list = self.encoder(x, ret_feature)
        else:
            h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        if ret_feature:
            return quant, emb_loss, info, phi_list
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant, x_prev, x_next):
        cond_dict = dict(
            phi_prev_list = self.encode(x_prev, ret_feature=True)[-1],
            phi_next_list = self.encode(x_next, ret_feature=True)[-1],
            frame_prev = F.pad(x_prev, (0, self.pad_w, 0, self.pad_h), mode='reflect'),
            frame_next = F.pad(x_next, (0, self.pad_w, 0, self.pad_h), mode='reflect')
        )
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, cond_dict)

        # check if image is padded and return the original part only
        if self.h_padded:
            dec = dec[:, :, 0:self.h0, :]
        if self.w_padded:
            dec = dec[:, :, :, 0:self.w0]
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, x_prev, x_next, return_pred_indices=False):

        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant, x_prev, x_next)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        x_prev = self.get_input(batch, 'prev_frame')
        x_next = self.get_input(batch, 'next_frame')
        xrec, qloss = self(x, x_prev, x_next)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        x_prev = self.get_input(batch, 'prev_frame')
        x_next = self.get_input(batch, 'next_frame')
        xrec, qloss = self(x, x_prev, x_next)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val"+suffix,
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x_prev = self.get_input(batch, 'prev_frame')
        x_next = self.get_input(batch, 'next_frame')
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x, x_prev, x_next)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x, x_prev, x_next)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    
class VQFlowNetInterface(VQFlowNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, x, ret_feature=False):
        '''
        Set ret_feature = True when encoding conditions in ddpm
        '''
        # Pad the input first so its size is deividable by 8.
        # this is to tolerate different f values, various size inputs, 
        # and some operations in the DDPM unet model.
        self.h0, self.w0 = x.shape[2:]
        # 8: window size for max vit
        # 2**(nr-1): f 
        # 4: factor of downsampling in DDPM unet
        min_side = 512#8 * 2**(self.encoder.num_resolutions-1) * 16
        min_side = min_side // 2 if self.h0 <= 256 else min_side
        if self.h0 % min_side != 0:
            pad_h = min_side - (self.h0 % min_side)
            if pad_h == self.h0: # this is to avoid padding 256 patches
                pad_h = 0
            x = F.pad(x, (0, 0, 0, pad_h), mode='reflect')
            self.h_padded = True
            self.pad_h = pad_h

        if self.w0 % min_side != 0:
            pad_w = min_side - (self.w0 % min_side)
            if pad_w == self.w0:
                pad_w = 0
            x = F.pad(x, (0, pad_w, 0, 0), mode='reflect')
            self.w_padded = True
            self.pad_w = pad_w

        phi_list = None
        if ret_feature:
            h, phi_list = self.encoder(x, ret_feature)
        else:
            h = self.encoder(x)
        h = self.quant_conv(h)
        if ret_feature:
            return h, phi_list
        return h

    def decode(self, h, x_prev, x_next, phi_prev_list, phi_next_list, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        cond_dict = dict(
            phi_prev_list = phi_prev_list,
            phi_next_list = phi_next_list,
            frame_prev = F.pad(x_prev, (0, self.pad_w, 0, self.pad_h), mode='reflect'),
            frame_next = F.pad(x_next, (0, self.pad_w, 0, self.pad_h), mode='reflect')
        )
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, cond_dict)

        # check if image is padded and return the original part only
        if self.h_padded:
            dec = dec[:, :, 0:self.h0, :]
        if self.w_padded:
            dec = dec[:, :, :, 0:self.w0]
        return dec