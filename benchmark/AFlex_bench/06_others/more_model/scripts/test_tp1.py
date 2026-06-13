import subprocess, os, sys, time

env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0'
log_path = '/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/logs/tp1_test.log'
os.makedirs(os.path.dirname(log_path), exist_ok=True)

cmd = ['/workspace/env/sglang-tier/bin/python', '-m', 'sglang.launch_server',
       '--model-path', '/models/Qwen3-30B-A3B/',
       '--tp', '1', '--host', '127.0.0.1', '--port', '40000',
       '--mem-fraction-static', '0.85',
       '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
       '--skip-server-warmup']

print(f"Starting server with cmd: {' '.join(cmd)}")
print(f"Log: {log_path}")

with open(log_path, 'w') as f:
    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"PID: {p.pid}")
    
    # Wait for completion or timeout
    try:
        ret = p.wait(timeout=60)
        print(f"Process exited with code: {ret}")
    except subprocess.TimeoutExpired:
        print("Still running after 60s (good, means model loaded)")
        # Check if healthy
        import requests
        try:
            r = requests.get("http://127.0.0.1:40000/health", timeout=5)
            print(f"Health: {r.status_code}")
        except:
            print("Health check pending...")
            
        # Try generate
        try:
            r = requests.post("http://127.0.0.1:40000/generate",
                            json={"text": "Hello", "sampling_params": {"max_new_tokens": 5, "temperature": 0}},
                            headers={"Content-Type": "application/json"},
                            timeout=30)
            print(f"Generate: {r.json()}")
        except Exception as e:
            print(f"Generate failed: {e}")
