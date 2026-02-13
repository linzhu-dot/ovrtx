# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import ovrtx
from PIL import Image

def main():
    # Create the Renderer and load a USD layer into it
    renderer = ovrtx.Renderer()
    renderer.add_usd("https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Robot-OVRTX/robot-ovrtx.usda")

    # Step the renderer to simulate the Camera at 60Hz
    products = renderer.step(
        render_products = {"/Render/Camera"},
        delta_time = 1.0 / 60
    )

    # Get the Camera output for the step as a numpy array and display it
    for _product_name, product in products.items():
        for frame in product.frames:
            with frame.render_vars["LdrColor"].map(device="cpu") as var:
                pixels = var.tensor.numpy()
                Image.fromarray(pixels).show()

if __name__ == "__main__":
    main()
