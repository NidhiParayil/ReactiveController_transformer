import gymnasium as gym
import numpy as np
import cv2
from robotics_transformer.sequence_agent import SequenceAgent
from robotics_transformer.tokenizers.image_tokenizer import RT1ImageTokenizer as ImageTokenizer
from robotics_transformer.sequence_agent_test_set_up import SequenceAgentTestSetUp
import tensorflow as tf
from tf_agents.trajectories import time_step as ts

# Initialize tokenizer and agent
image_tokenizer = ImageTokenizer(embedding_output_dim=512)

setup = SequenceAgentTestSetUp()
setup.setUp()  # ⬅️ This is what tf.test.main() would normally call

setup.sequence_agent_cls = SequenceAgent  # ← manually assign it since setUp() won't be called
agent = setup.create_agent_and_initialize()


def rt1_policy(image):
    img = image.astype(np.float32) / 255.0
    img = np.resize(img, (256, 320, 3))  # RT-1 expects this shape
    batched = img[np.newaxis, np.newaxis, ...]  # shape: (B=1, T=1, H, W, 3)

    obs = {
        "image": tf.convert_to_tensor(batched, dtype=tf.float32),
        "natural_language_embedding": tf.zeros((1, 1, 512), dtype=tf.float32),
    }

    time_step = ts.restart(obs)
    action = agent.policy.action(time_step).action
    return {k: v.numpy().squeeze() for k, v in action.items()}

# Initialize env with render_mode
env = gym.make("HalfCheetah-v4", render_mode="rgb_array")
obs, info = env.reset(seed=42)

# Get action space dimension
action_dim = env.action_space.shape[0]

for step in range(200):
    # Get rendered image (as RGB numpy array)
    frame = env.render()

    # Resize to match RT-1 input expectations (e.g., 224x224)
    resized_frame = cv2.resize(frame, (224, 224))

    # Feed through dummy RT-1 policy
    action = rt1_policy(resized_frame)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Show the sim (cv2 window)
    cv2.imshow("MuJoCo with RT-1", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if terminated or truncated:
        obs, info = env.reset()

env.close()
cv2.destroyAllWindows()
