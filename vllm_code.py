from vllm import LLM, SamplingParams
prompts = ['what is life']*100
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="tiiuae/falcon-7b", quantization='awq')
outputs = llm.generate(prompts, sampling_params)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
