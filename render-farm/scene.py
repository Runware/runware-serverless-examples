"""Builds the scene and renders one frame. Run by Blender, not by the server.

Procedural on purpose: there is no .blend file in this example, so there is
nothing to keep in step with the code that renders it.
"""

import math
import sys

import bpy

FRAMES = 240


def device() -> str:
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        preferences.compute_device_type = kind
        preferences.get_devices()
        usable = [d for d in preferences.devices if d.type == kind]
        if usable:
            for d in preferences.devices:
                d.use = d.type == kind
            return "GPU"

    return "CPU"


def build() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    material = bpy.data.materials.new("metal")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.6, 0.7, 0.75, 1.0)
    shader.inputs["Metallic"].default_value = 0.9
    shader.inputs["Roughness"].default_value = 0.25

    for index in range(6):
        angle = index / 6 * math.tau
        bpy.ops.mesh.primitive_torus_add(
            major_radius=1.1,
            minor_radius=0.28,
            location=(math.cos(angle) * 1.8, math.sin(angle) * 1.8, 0.4),
            rotation=(math.pi / 2, 0, angle),
        )
        bpy.context.object.data.materials.append(material)

    bpy.ops.mesh.primitive_plane_add(size=40)
    bpy.ops.object.light_add(type="AREA", location=(4, -4, 7))
    bpy.context.object.data.energy = 2000

    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera


def place_camera(frame: int) -> None:
    angle = (frame % FRAMES) / FRAMES * math.tau
    camera = bpy.context.scene.camera
    camera.location = (math.cos(angle) * 9, math.sin(angle) * 9, 4.5)
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = (math.radians(68), 0, angle + math.pi / 2)


def main() -> None:
    frame, samples, width, height, output = (
        int(sys.argv[-5]), int(sys.argv[-4]), int(sys.argv[-3]),
        int(sys.argv[-2]), sys.argv[-1],
    )

    build()
    place_camera(frame)

    scene = bpy.context.scene
    used = device()
    scene.cycles.device = used
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = output

    print(f"RUNWARE_DEVICE={used}", flush=True)
    bpy.ops.render.render(write_still=True)


main()
