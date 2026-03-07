# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import sys

import ovrtx
from PIL import Image

USD_URL = "https://omniverse-content-production.s3.us-west-2.amazonaws.com/Samples/Robot-OVRTX/robot-ovrtx.usda"

def main():
    # [snippet:create-renderer]
    # Create the Renderer and load a USD layer into it
    print("Creating renderer. The first run of the application will take some time as shaders are compiled and cached...", file=sys.stderr)
    renderer = ovrtx.Renderer()
    print("Renderer created.", file=sys.stderr)
    # [/snippet:create-renderer]

    # [snippet:add-usd]
    print(f"Adding {USD_URL} at root...", file=sys.stderr)
    renderer.add_usd(USD_URL)
    print("USD loaded.", file=sys.stderr)
    # [/snippet:add-usd]

    # [snippet:step]
    # Step the renderer to simulate the Camera at 60Hz
    print("Stepping renderer...", file=sys.stderr)
    products = renderer.step(
        render_products = {"/Render/Camera"},
        delta_time = 1.0 / 60
    )
    print("Stepped renderer.", file=sys.stderr)
    # [/snippet:step]

    # [snippet:read-render-output]
    # Get the Camera output for the step as a numpy array and display it
    print("Fetching results...", file=sys.stderr)
    for _product_name, product in products.items():
        for frame in product.frames:
            with frame.render_vars["LdrColor"].map(device=ovrtx.Device.CPU) as var:
                pixels = var.tensor.numpy()
                Image.fromarray(pixels).show()
    print("Fetched results.", file=sys.stderr)
    # [/snippet:read-render-output]

if __name__ == "__main__":
    main()
