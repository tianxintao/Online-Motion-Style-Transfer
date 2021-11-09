import sys
sys.path.append('./')
sys.path.append('../')
import bpy
import numpy as np

from options import Options
from load_bvh import load_bvh
from scene import make_scene, add_material_for_character, add_rendering_parameters

if __name__ == '__main__':
    args = Options(sys.argv).parse()

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # set default color
    input_color = (0.8, 0.8, 0.2, 1)
    ours_color = (0.15, 0.6, 0, 1)
    aberman_color = (0.1, 0.6, 0.6, 1)
    park_color = (0.3, 0.3, 0.8, 1)
    ablated_color = (0.7, 0.0, 0.0, 1)
    
    # neutral to angry run
    # file_name = 'bvh/neutral_to_angry_run/neutral_to_original_run_24.bvh'
    # file_name = 'bvh/neutral_to_angry_run/neutral_to_angry_run_24.bvh'
    # file_name = 'bvh/neutral_to_angry_run/neutral_to_angry_run_aberman.bvh'
    # file_name = 'bvh/neutral_to_angry_run/neutral_to_angry_run_park.bvh'
    # starting_frame_array = [0, 12, 24, 36, 48]
    # camera_position = (10, -39.547, 14.762)
    # camera_rotation = tuple(np.array([76.4, 0.63, 11.1]) / 180 * 3.1415)
    # light_position=(0, 0, 20)
    
    # neutral to sexy kick
    # file_name = 'bvh/neutral_to_sexy_kick_comparison/neutral_to_original_kick_27.bvh'
    # file_name = 'bvh/neutral_to_sexy_kick_comparison/neutral_to_sexy_kick_27.bvh'
    # file_name = 'bvh/neutral_to_sexy_kick_comparison/aberman_sexy_kick.bvh'
    # file_name = 'bvh/neutral_to_sexy_kick_comparison/park_sexy_kick.bvh'
    # starting_frame_array = [0, 12, 24, 36, 48]
    # camera_position = (35.43, 9.793, 14.762)
    # camera_rotation = tuple(np.array([74.3, 0.0, 89]) / 180 * 3.1415)
    # light_position=(10, 0, 20)
    
    # neutral to childlike jump
    # file_name = 'bvh/neutral_to_childlike_jump/neutral_to_original_jump_25.bvh'
    # file_name = 'bvh/neutral_to_childlike_jump/neutral_to_childlike_jump_25.bvh'
    # file_name = 'bvh/neutral_to_childlike_jump/aberman_childlike_jump.bvh'
    # file_name = 'bvh/neutral_to_childlike_jump/park_childlike_jump.bvh'
    # starting_frame_array = [0, 12, 24, 36, 48, 60]
    # camera_position = (21.5, -39.547, 14.762)
    # camera_rotation = tuple(np.array([77, 0.63, 17.5]) / 180 * 3.1415)
    # light_position=(0, 0, 20)
    
    # first ten frames: neutral to childlike jump
    # file_name = 'bvh/tenframe_neutral_to_proud_walk/ours.bvh'
    # file_name = 'bvh/tenframe_neutral_to_proud_walk/aberman.bvh'
    # starting_frame_array = [0, 2, 4, 6, 8]
    # camera_position = (10, -39.547, 14.762)
    # camera_rotation = tuple(np.array([76.4, 0.63, 11.1]) / 180 * 3.1415)
    # light_position=(0, 0, 20)
    
    # ablation: both supervision modules
    file_name = 'bvh/abalation/reconstruction/neutral_to_proud_walk_21.bvh'
    # file_name = 'bvh/abalation/reconstruction/neutral_to_proud_walk_21_abalation.bvh'
    starting_frame_array = [0, 5, 10, 15, 20, 25]
    camera_position = (10, -39.547, 14.762)
    camera_rotation = tuple(np.array([76.4, 0.63, 11.1]) / 180 * 3.1415)
    light_position=(0, 0, 20)
    
    # ablation: both attention mechanism
    # file_name = 'bvh/abalation/no_attention/neutral_to_proud_jump_25.bvh'
    # file_name = 'bvh/abalation/no_attention/neutral_to_proud_jump_ablation_25.bvh'
    # starting_frame_array = [0, 12, 24, 36, 48, 60]
    # camera_position = (21.5, -39.547, 14.762)
    # camera_rotation = tuple(np.array([77, 0.63, 17.5]) / 180 * 3.1415)
    # light_position=(0, 0, 20)
    
    # ablation: learnt initial states
    # file_name = 'bvh/abalation/initial_state/neutral_to_childlike_punch_26.bvh'
    # file_name = 'bvh/abalation/initial_state/initial_state_ablation.bvh'
    # starting_frame_array = [0, 8, 16, 24, 32, 40]
    # camera_position = (21.5, -39.547, 14.762)
    # camera_rotation = tuple(np.array([77, 0.63, 17.5]) / 180 * 3.1415)
    # light_position=(0, 0, 20)
    
    # ablation: no perceptual loss
    # file_name = 'bvh/neutral_to_angry_run/neutral_to_angry_run_24.bvh'
    # file_name = 'bvh/abalation/no_perceptual_loss/neutral_to_angry_run_ablation_24.bvh'
    # starting_frame_array = [0, 12, 24, 36, 48]
    # camera_position = (10, -39.547, 14.762)
    # camera_rotation = tuple(np.array([76.4, 0.63, 11.1]) / 180 * 3.1415)
    # light_position=(0, 0, 20)
    

    scene = make_scene(camera_position=camera_position, camera_rotation=camera_rotation, light_position=light_position)

    

    for index in range(len(starting_frame_array)):
        character = load_bvh(file_name, index=index, start=starting_frame_array[index])
        add_material_for_character(character, color=ours_color)

    bpy.ops.object.select_all(action='DESELECT')

    add_rendering_parameters(bpy.context.scene, args, scene[1])

    if args.render:
        bpy.ops.render.render(animation=True, use_viewport=True)
