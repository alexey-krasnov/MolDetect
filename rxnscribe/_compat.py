"""
Compatibility shim for transformers >= 4.49 with the old timm this project is pinned to.

transformers 4.49 added the `timm_wrapper` model type. Its configuration module does
    from timm.data import ImageNetInfo, infer_imagenet_subset
whenever timm is installed, but timm <= 0.5.4 (pinned: the MolScribe / MolDetect Swin
checkpoints need it) has neither name. MolDetect's decoder is built with
`EncoderDecoderModel`, which scans ALL registered model types, so merely creating the
model fails with "cannot import name 'ImageNetInfo' from 'timm.data'".

Nothing here uses TimmWrapper, so two inert stand-ins are enough. They raise if anyone
ever tries to really use them, instead of silently doing the wrong thing.
Remove this file when timm is upgraded (or transformers' timm import is guarded).
"""


def _install():
    try:
        import timm.data as timm_data
    except ImportError:          # timm not installed: transformers does not import it either
        return

    def _unsupported(*args, **kwargs):
        raise NotImplementedError(
            "timm.data.ImageNetInfo / infer_imagenet_subset are stand-ins for the installed "
            "timm (too old for transformers' TimmWrapper); upgrade timm to use TimmWrapper.")

    if not hasattr(timm_data, "ImageNetInfo"):
        timm_data.ImageNetInfo = type("ImageNetInfo", (), {"__init__": _unsupported})
    if not hasattr(timm_data, "infer_imagenet_subset"):
        timm_data.infer_imagenet_subset = _unsupported


_install()
