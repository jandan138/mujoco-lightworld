import gymnasium as gym
import time
import sys

def test_visualization():
    print("1. Initializing environment with render_mode='human'...")
    try:
        # 尝试以 human 模式初始化，这会触发窗口创建
        env = gym.make("Walker2d-v4", render_mode="human")
        print("   [SUCCESS] Environment initialized.")
    except Exception as e:
        print(f"   [FAILED] Environment initialization failed: {e}")
        return

    print("2. Resetting environment...")
    try:
        obs, _ = env.reset()
        print("   [SUCCESS] Environment reset. Model loaded.")
    except Exception as e:
        print(f"   [FAILED] Environment reset failed: {e}")
        return

    print("3. Starting simulation loop (Press Ctrl+C to stop)...")
    try:
        for i in range(100):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 渲染一帧
            env.render()
            
            if i % 10 == 0:
                print(f"   Step {i}: Action sampled, render call successful.")
            
            time.sleep(0.01)  # 控制帧率
            
            if terminated or truncated:
                env.reset()
                
        print("   [SUCCESS] Simulation loop completed.")
        
    except KeyboardInterrupt:
        print("\n   [INFO] User interrupted.")
    except Exception as e:
        print(f"   [FAILED] Simulation error: {e}")
    finally:
        env.close()
        print("4. Environment closed.")

if __name__ == "__main__":
    test_visualization()
