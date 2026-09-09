"""Format readers: what a model file says about itself.

One module per on-disk format, each reading only the header. Nothing in
here opens a tensor, loads a weight, or imports a tensor library.
"""
