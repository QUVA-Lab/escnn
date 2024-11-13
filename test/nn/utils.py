from unittest import TestCase

import torch
import io

def check_torch_load_save(tester: TestCase, mod: torch.nn.Module):
    buffer = io.BytesIO()

    torch.save(mod, buffer)

    buffer.seek(0)

    mod_load_save = torch.load(buffer)

    state = mod.state_dict()
    state_load_save = mod_load_save.state_dict()

    tester.assertEqual(state.keys(), state_load_save.keys())

    for k in state:
        torch.testing.assert_close(state_load_save[k], state[k])

