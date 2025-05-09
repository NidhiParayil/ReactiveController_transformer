import numpy as np
import cv2
import tensorflow as tf
import tensorflow_probability as tfp  # ✅ Add this line!
from tf_agents.trajectories import time_step as ts
from mujoco_env import RT1LikeMuJoCoEnv
import tensorflow_hub as hub


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embedding = embed(["move to the red sphere"]).numpy().squeeze()

policy = tf.saved_model.load("robotics_transformer/trained_checkpoints/rt1simreal")
print("✅ Model loaded successfully!")

# Initialize environment
env = RT1LikeMuJoCoEnv(render_mode="rgb_array")
obs, _ = env.reset()



for step in range(200):
    frame = env._get_obs()["image"]
    img = cv2.resize(frame, (320, 256)).astype(np.uint8)

    # Create input dict as expected by SavedModel
    inputs = {
        "0/observation/image": tf.convert_to_tensor([img], dtype=tf.uint8),
        "0/observation/natural_language_embedding": tf.convert_to_tensor([embedding], dtype=tf.float32),

        # Minimal valid stubs
        "0/observation/vector_to_go": tf.zeros([1, 3], dtype=tf.float32),
        "0/observation/src_rotation": tf.zeros([1, 4], dtype=tf.float32),
        "0/observation/natural_language_instruction": tf.convert_to_tensor([""], dtype=tf.string),
        "0/observation/gripper_closed": tf.zeros([1, 1], dtype=tf.float32),
        "0/observation/gripper_closedness_commanded": tf.zeros([1, 1], dtype=tf.float32),
        "0/observation/height_to_bottom": tf.zeros([1, 1], dtype=tf.float32),
        "0/observation/rotation_delta_to_go": tf.zeros([1, 3], dtype=tf.float32),
        "0/observation/workspace_bounds": tf.zeros([1, 3, 3], dtype=tf.float32),
        "0/observation/base_pose_tool_reached": tf.zeros([1, 7], dtype=tf.float32),
        "0/observation/orientation_box": tf.zeros([1, 2, 3], dtype=tf.float32),
        "0/observation/orientation_start": tf.zeros([1, 4], dtype=tf.float32),
        "0/observation/robot_orientation_positions_box": tf.zeros([1, 3, 3], dtype=tf.float32),

        "0/step_type": tf.zeros([1], dtype=tf.int32),
        "0/reward": tf.zeros([1], dtype=tf.float32),
        "0/discount": tf.ones([1], dtype=tf.float32),

        # Past trajectory state (stubbed)
        "1/action_tokens": tf.zeros([1, 6, 11, 1, 1], dtype=tf.int32),
        "1/image": tf.zeros([1, 6, 256, 320, 3], dtype=tf.uint8),
        "1/step_num": tf.zeros([1, 1, 1, 1, 1], dtype=tf.int32),
        "1/t": tf.zeros([1, 1, 1, 1, 1], dtype=tf.int32),
    }

    # Call SavedModel
    outputs = policy.signatures["action"](**inputs)
    world_vector = outputs["action/world_vector"].numpy().squeeze()

    print("RT-1 action:", world_vector)
    obs, reward, terminated, truncated, _ = env.step(world_vector)

    cv2.imshow("RT-1 MuJoCo", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
cv2.destroyAllWindows()