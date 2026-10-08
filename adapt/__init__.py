"""The public RTA adaptation entry point."""

import inspect

from .rta import RTA


METHOD_CLASSES = {"rta": RTA}


def get_method(args, device):
    """Build the single method exposed by the clean paper implementation."""
    if args.method not in METHOD_CLASSES:
        raise ValueError(f"Unknown method: {args.method!r}; only 'rta' is supported")

    method_class = METHOD_CLASSES[args.method]
    class_args = {
        parameter.name
        for parameter in inspect.signature(method_class.__init__).parameters.values()
        if parameter.name != "self"
    }
    method_args = {
        key: value for key, value in vars(args).items() if key in class_args
    }
    method_args["device"] = device
    method_args["classes"] = args.classes

    if args.patch_size is not None:
        method_args["patch_size"] = tuple(args.patch_size)
    else:
        method_args["patch_size"] = (224, 224)

    if method_args.get("patch_stride") is not None and method_args["patch_stride"] <= 0:
        raise ValueError(f"patch_stride must be positive, got {method_args['patch_stride']}")

    required = [
        parameter.name
        for parameter in inspect.signature(method_class.__init__).parameters.values()
        if parameter.name != "self" and parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in required if name not in method_args]
    if missing:
        raise ValueError(f"Missing arguments for {method_class.__name__}: {missing}")

    print("\nMethod +++++++++++++++++++++++++++++++++++++")
    print(f"Selected Method: {method_class.__name__}")
    print(f"TTA enabled: {args.adapt}")
    print(f"SAM model: {method_args.get('sam_model_type', 'vit_h')}")
    print(f"Reliability bands: 0-{method_args.get('percentile_low', 40)}%, "
          f"{method_args.get('percentile_low', 40)}-{method_args.get('percentile_high', 85)}%, "
          f"{method_args.get('percentile_high', 85)}-100%")
    print("----------------------------------------")

    return method_class(**method_args)
