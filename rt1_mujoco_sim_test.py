import numpy as np
import cv2
import tensorflow as tf
from tf_agents.trajectories import time_step as ts
from robotics_transformer.sequence_agent import SequenceAgent
from robotics_transformer.tokenizers.image_tokenizer import RT1ImageTokenizer as ImageTokenizer
from robotics_transformer.sequence_agent_test_set_up import SequenceAgentTestSetUp
from mujoco_env import RT1LikeMuJoCoEnv  # ⬅️ custom env
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embedding = embed(["move to the red sphere"]).numpy()  # shape (1, 512)
# Initialize tokenizer and agent
image_tokenizer = ImageTokenizer(embedding_output_dim=512)


setup = SequenceAgentTestSetUp()
setup.setUp()
setup.sequence_agent_cls = SequenceAgent
agent = setup.create_agent_and_initialize()

def rt1_policy(image):
    img = image.astype(np.float32) / 255.0
    img = np.resize(img, (256, 320, 3))  # expected input shape
    batched = img[np.newaxis, np.newaxis, ...]

    obs = {
        "image": tf.convert_to_tensor(batched, dtype=tf.float32),
        "natural_language_embedding": tf.convert_to_tensor(embedding[np.newaxis, ...], dtype=tf.float32)
    }

    time_step = ts.restart(obs)
    action = agent.policy.action(time_step).action
    return {k: v.numpy().squeeze() for k, v in action.items()}

# Initialize custom MuJoCo env
env = RT1LikeMuJoCoEnv(render_mode="rgb_array")
obs, _ = env.reset()

for step in range(200):
    frame = env._get_obs()["image"]
    resized_frame = cv2.resize(frame, (224, 224))
    action = rt1_policy(resized_frame)

    # Replace with actual control fields (adjust as needed)
    ctrl_action = np.array(action.get("world_vector", [0.0, 0.0, 0.0]))
    obs, reward, terminated, truncated, _ = env.step(ctrl_action)

    cv2.imshow("MuJoCo + RT-1", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
cv2.destroyAllWindows()
