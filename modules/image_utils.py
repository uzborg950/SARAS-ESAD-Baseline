# from https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/data/transforms/transforms.py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
def get_size(image_size, min_size, max_size):
    if min_size == max_size:

        return (min_size, max_size)

    else:
        w, h = image_size
        size = min_size
        max_size = max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:  # 1920/1080  * 200
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(round(size * h / w))
        else:
            oh = size
            ow = int(round(size * w / h))

        return (ow, oh)