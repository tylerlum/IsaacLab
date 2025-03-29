from isaaclab_rl.rl_games import RlGamesVecEnvWrapper
from omegaconf import OmegaConf
from typing import Any


# Same wrapper as RL-Games
class SimpleRlVecEnvWrapper(RlGamesVecEnvWrapper):
    pass


def match_value(key: str, *cases) -> Any:
    """
    Return a corresponding result for the given `key` by matching it
    against pairs in `cases` (each pair is (possible_key, result_value)) and a default_value.

    If no match is found in `(possible_key, result_value)`, return `default_value`.

    Args:
        key: The key to match on.
        *cases: A sequence of alternating (match_key, result_value) pairs and a default_value.

    Returns:
        The first matching result_value in `cases`, or `default_value` otherwise.

    Raises:
        AssertionError: If the `cases` argument doesn't have an even number of items.
    """
    # We expect pairs, so `cases` must have even length
    assert len(cases) % 2 == 1, (
        f"Expected an odd number of arguments for pairs + default_value, got {len(cases)}: {cases}. key: {key}"
    )

    n_pairs = len(cases) // 2
    possible_keys = [cases[2*i] for i in range(n_pairs)]
    result_values = [cases[2*i + 1] for i in range(n_pairs)]
    default_value = cases[-1]

    # Go through each pair: (possible_key, result_value)
    for possible_key, result_value in zip(possible_keys, result_values):
        if key == possible_key:
            return result_value

    # If no match, return the default
    return default_value

def add_omegaconf_resolvers() -> None:
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver("match_value", match_value)
    OmegaConf.register_new_resolver("eval", eval)

