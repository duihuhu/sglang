from sglang import Engine

def main():
 engine=Engine(model_path="/models/Qwen3-30B-A3B",tp_size=8,ep_size=8,moe_runner_backend="triton",moe_a2a_backend="none",disable_cuda_graph=True,disable_custom_all_reduce=True,max_total_tokens=8192)
 out=engine.generate(prompt="The capital of France is",sampling_params={"temperature":0,"max_new_tokens":4},return_logprob=True,top_logprobs_num=20)
 print(out)
 engine.shutdown()
if __name__=="__main__":main()
