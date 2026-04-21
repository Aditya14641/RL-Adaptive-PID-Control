from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = 'logs/CS1/CS1_SystemSimplePID_PPO_AR_True_2_use_sde_False_ES_True_extra_BetterES_SystemFix'
print(f"Scanning folder: {log_dir}")

try:
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tags = ea.Tags()
    print("\n--- FOUND SCALAR TAGS ---")
    for tag in tags.get('scalars', []):
        print(f" - {tag}")
except Exception as e:
    print(f"Error reading TensorBoard file: {e}")
