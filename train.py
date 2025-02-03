import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
import model
import environment

if __name__ == "__main__":
    env = environment.Environment(composition_depth = 3, n_unique_elements = 5, n_syms = 10, 
                      n_winstates = 10, max_element_length = 3)

    lm = model.Model("Qwen/Qwen2.5-0.5B")
    
    ppo_config = PPOConfig(
        batch_size = 4,           
        forward_batch_size = 2,   # how many samples per forward pass
        lr = 1e-5,             # learning rate
        log_with = "wandb", 
    )

    # set up PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=lm.model,
        tokenizer=lm.tokenizer,
    )

    # 4) Define an environment loop / single training step
    for epoch in range(num_epochs):
        # a) Prompt your model to get a batch of responses
        prompts = [
            "Explain why reinforcement learning is useful.",
            "What is the meaning of life?",
            # ...
        ]
        batch = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = batch["input_ids"]

        # Generate responses
        with torch.no_grad():
            response_ids = ppo_trainer.model.generate(
                input_ids=input_ids.cuda(),
                max_new_tokens=50
            )

    # b) Compute a reward for each response
    #    (Your custom function here.)
    all_rewards = []
    for i, resp_ids in enumerate(response_ids):
        response_str = tokenizer.decode(resp_ids[input_ids.shape[1]:], skip_special_tokens=True)
        
        # Provide a custom reward. For instance:
        # reward = your_reward_function(response_str)
        reward = your_reward_function(response_str)

        all_rewards.append(reward)

    # c) Run a PPO update
    #    You have to pass in "query" tokens (the prompt) and "response" tokens
    #    separately for the trainer to do the PPO step.
    #    Typically, the "response" is the newly generated tokens after the prompt.
    #    So we separate out from input_ids (the prompts) and response_ids (the completions).
    # 
    #    ppo_trainer.step() expects lists (or batched) of strings or token ids + rewards.

    # Convert each prompt to string (if needed)
    prompt_texts = [prompts[i] for i in range(len(prompts))]
    response_texts = []
    for i, resp_ids in enumerate(response_ids):
        response_texts.append(
            tokenizer.decode(resp_ids[input_ids.shape[1]:], skip_special_tokens=True)
        )

    # Now pass them to the PPO trainer
    train_stats = ppo_trainer.step(prompt_texts, response_texts, all_rewards)
    # Logging or debugging
    print(f"Epoch {epoch}: {train_stats}")
