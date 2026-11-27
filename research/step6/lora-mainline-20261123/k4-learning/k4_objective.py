"""Experimental ARX K4 supervision; no changes to inference or execution.

The forward pass is copied from pinned OpenPI f4e52e6 pi0.compute_loss.
The control has auxiliary_weight=0 and retains the original weighted flow loss.
"""
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as _model
from openpi.models.pi0 import make_attn_mask


def flow_and_clean(pi, rng, observation, actions, train=True):
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001
    t = time[..., None, None]
    x_t = t * noise + (1 - t) * actions
    prefix, prefix_mask, prefix_ar = pi.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, cond = pi.embed_suffix(observation, x_t, time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
    (_, output), _ = pi.PaliGemma.llm(
        [prefix, suffix], mask=make_attn_mask(mask, ar),
        positions=jnp.cumsum(mask, axis=1)-1, adarms_cond=[None, cond])
    velocity = pi.action_out_proj(output[:, -pi.action_horizon:])
    return jnp.mean((velocity - (noise-actions))**2, axis=-1), x_t-t*velocity


from k4_fk import make_fk


def make_auxiliary(arms, stats):
    fks = [make_fk(arm) for arm in arms]
    low = jnp.asarray(stats['actions']['q01'], dtype=jnp.float32)
    scale = jnp.asarray(stats['actions']['q99'], dtype=jnp.float32)-low+1e-6
    arm_mask = jnp.asarray([1]*6+[0]+[1]*6+[0], dtype=jnp.float32)

    def decode(actions, state):
        return (actions[..., :14]+1)*.5*scale+low+state[:, None, :]*arm_mask

    def auxiliary(prediction, target, raw_state, valid_rows):
        pred, goal = decode(prediction[:, :4], raw_state), decode(target[:, :4], raw_state)
        valid = jnp.arange(4)[None, :] < valid_rows[:, None]
        def mean_valid(value):
            return jnp.sum(value*valid)/jnp.maximum(jnp.sum(valid), 1)
        position, rotation = 0., 0.
        for side, fk in enumerate(fks):
            actual = fk(pred[..., side*7:side*7+6])
            expected = fk(goal[..., side*7:side*7+6])
            position += mean_valid(jnp.sum(((actual[..., :3, 3]-expected[..., :3, 3])/.05)**2, axis=-1))*.5
            # Chordal distance approximates squared angle near zero; avoids acos derivative singularity.
            rotation += mean_valid(jnp.sum((actual[..., :3, :3]-expected[..., :3, :3])**2, axis=(-2, -1))/(2*.35**2))*.5
        grip = mean_valid(jnp.mean((pred[..., jnp.array([6, 13])]-goal[..., jnp.array([6, 13])])**2, axis=-1))
        return position+rotation+grip, jnp.array([position, rotation, grip])
    return auxiliary
