
import os



# Given root path of 'Driving' sceneflow dataset this returns list of tuples
# (image_a, image_b, flow_forwards a->b, flow_backwards b->a)
# filenames.
def get_filenames_for_sceneflow_driving(scene_flow_root):
    driving_path = os.path.join(scene_flow_root, 'Driving')
    filenames = []

    for fl_dir in ['15mm_focallength', '35mm_focallength']:
        for fb_dir in ['scene_backwards', 'scene_forwards']:
            for fs_dir in ['fast', 'slow']:
                for lr_dir in ['left', 'right']:
                    dir_path = os.path.join(driving_path, 'frames_finalpass', fl_dir, fb_dir, fs_dir, lr_dir)
                    image_filenames = [filename for filename in os.listdir(dir_path) if filename.endswith('.png')]
                    image_filenames.sort()
                    rel_path = os.path.relpath(dir_path, scene_flow_root)
                    image_filenames = [os.path.join(rel_path, filename) for filename in image_filenames]

                    dir_path = os.path.join(driving_path, 'optical_flow', fl_dir, fb_dir, fs_dir, 'into_future', lr_dir)
                    flow_forward_filenames = [filename for filename in os.listdir(dir_path) if
                                              filename.endswith('.pfm')]
                    flow_forward_filenames.sort()
                    rel_path = os.path.relpath(dir_path, scene_flow_root)
                    flow_forward_filenames = [os.path.join(rel_path, filename) for filename in flow_forward_filenames]

                    dir_path = os.path.join(driving_path, 'optical_flow', fl_dir, fb_dir, fs_dir, 'into_past', lr_dir)
                    flow_backwards_filenames = [filename for filename in os.listdir(dir_path) if
                                                filename.endswith('.pfm')]
                    flow_backwards_filenames.sort()
                    rel_path = os.path.relpath(dir_path, scene_flow_root)
                    flow_backwards_filenames = [os.path.join(rel_path, filename) for filename in
                                                flow_backwards_filenames]

                    filenames.extend(zip(image_filenames[:-1], image_filenames[1:], flow_forward_filenames[:-1],
                                         flow_backwards_filenames[1:]))

    return filenames


# Given root path of 'Monkaa' sceneflow dataset this returns list of tuples
# (image_a, image_b, flow_forwards a->b, flow_backwards b->a)
# filenames.
def get_filenames_for_sceneflow_monkaa(scene_flow_root):
    monkaa_path = os.path.join(scene_flow_root, 'Monkaa')
    filenames = []

    for fl_dir in ['a_rain_of_stones_x2', 'eating_camera2_x2', 'eating_naked_camera2_x2', 'eating_x2', 'family_x2',
                       'flower_storm_augmented0_x2', 'flower_storm_augmented1_x2', 'flower_storm_x2',
                       'funnyworld_augmented0_x2', 'funnyworld_augmented1_x2', 'funnyworld_camera2_augmented0_x2',
                       'funnyworld_camera2_augmented1_x2', 'funnyworld_camera2_x2', 'funnyworld_x2',
                       'lonetree_augmented0_x2', 'lonetree_augmented1_x2', 'lonetree_difftex2_x2',
                       'lonetree_difftex_x2', 'lonetree_winter_x2', 'lonetree_x2', 'top_view_x2',
                       'treeflight_augmented0_x2', 'treeflight_augmented1_x2', 'treeflight_x2']:
        for lr_dir in ['left', 'right']:
            dir_path = os.path.join(monkaa_path, 'frames_finalpass', fl_dir, lr_dir)
            image_filenames = [filename for filename in os.listdir(dir_path) if filename.endswith('.png')]
            image_filenames.sort()
            rel_path = os.path.relpath(dir_path, scene_flow_root)
            image_filenames = [os.path.join(rel_path, filename) for filename in image_filenames]

            dir_path = os.path.join(monkaa_path, 'optical_flow', fl_dir, 'into_future', lr_dir)
            flow_forward_filenames = [filename for filename in os.listdir(dir_path) if
                                      filename.endswith('.pfm')]
            flow_forward_filenames.sort()
            rel_path = os.path.relpath(dir_path, scene_flow_root)
            flow_forward_filenames = [os.path.join(rel_path, filename) for filename in flow_forward_filenames]

            dir_path = os.path.join(monkaa_path, 'optical_flow', fl_dir, 'into_past', lr_dir)
            flow_backwards_filenames = [filename for filename in os.listdir(dir_path) if
                                        filename.endswith('.pfm')]
            flow_backwards_filenames.sort()
            rel_path = os.path.relpath(dir_path, scene_flow_root)
            flow_backwards_filenames = [os.path.join(rel_path, filename) for filename in
                                        flow_backwards_filenames]

            filenames.extend(zip(image_filenames[:-1], image_filenames[1:], flow_forward_filenames[:-1],
                                 flow_backwards_filenames[1:]))

    return filenames

if __name__ == '__main__':
    filenames = get_filenames_for_sceneflow_driving('/home/tp/src/datasets/SceneFlow')
    print('Driving samples:', len(filenames))
    print('Like:', filenames[0])

    filenames = get_filenames_for_sceneflow_monkaa('/home/tp/src/datasets/SceneFlow')
    print('Monkaa samples:', len(filenames))
    print('Like:', filenames[0])
