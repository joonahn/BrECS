from models.unet import Res16UNet14C, Res16UNet18B, Res16UNet34C, Res16UNet34NoSkipC, Res16UNet34NoSkipNoScaleC

BACKBONES = {
    Res16UNet14C.name: Res16UNet14C,
    Res16UNet18B.name: Res16UNet18B,
    Res16UNet34C.name: Res16UNet34C,
    Res16UNet34NoSkipC.name: Res16UNet34NoSkipC,
    Res16UNet34NoSkipNoScaleC.name: Res16UNet34NoSkipNoScaleC,
}