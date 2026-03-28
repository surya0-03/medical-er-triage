from my_env.server.environment import MedicalEmergencyRoomEnv
from my_env.models import Action

print('=== Medical ER Triage OpenEnv Demo ===')

for difficulty in ['easy', 'medium', 'hard']:
    print(f'\n--- Task: {difficulty} ---')
    env = MedicalEmergencyRoomEnv(difficulty=difficulty, seed=42)
    obs = env.reset()
    print(f'Reset: {len(obs.patients)} patients, queue={len(obs.waiting_queue)}')

    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        mask = env.get_action_mask()
        if mask['assign_esi_patient_ids']:
            pid = mask['assign_esi_patient_ids'][0]
            action = Action(action_type='assign_esi', patient_id=pid, esi_level=3)
        elif mask['allocate_bed']:
            pid = list(mask['allocate_bed'].keys())[0]
            beds = mask['allocate_bed'][pid]
            bed = 'icu' if 'icu' in beds else 'general' if 'general' in beds else 'hallway'
            action = Action(action_type='allocate_bed', patient_id=pid, bed_type=bed)
        else:
            action = Action(action_type='divert')

        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        steps += 1

    print(f'Steps: {steps}, Total reward: {round(total_reward, 4)}, Avg: {round(total_reward/steps, 4)}')

print('\nDemo complete.')

