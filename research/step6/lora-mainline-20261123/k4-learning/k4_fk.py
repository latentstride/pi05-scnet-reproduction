"""Differentiable ARX FK and regression against the existing audited chain."""
import jax
import jax.numpy as jnp
import numpy as np

def make_fk(arm):
    frames = [(jnp.asarray(f, dtype=jnp.float32), None if a is None else np.asarray(a))
              for f, a in arm.chain.joints]
    tool = jnp.asarray(arm.offset(), dtype=jnp.float32)

    def fk(q):
        transform = jnp.broadcast_to(jnp.eye(4), q.shape[:-1]+(4, 4))
        joint = 0
        for frame, axis in frames:
            transform = jnp.matmul(transform, frame, precision=jax.lax.Precision.HIGHEST)
            if axis is not None:
                x, y, z = axis
                cross = jnp.asarray([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                angle = q[..., joint, None, None]
                rotation = jnp.eye(3)+jnp.sin(angle)*cross+(1-jnp.cos(angle))*(cross@cross)
                transform = transform.at[..., :3, :3].set(jnp.matmul(
                    transform[..., :3, :3], rotation, precision=jax.lax.Precision.HIGHEST))
                joint += 1
        return jnp.matmul(transform, tool, precision=jax.lax.Precision.HIGHEST)
    return fk


def verify_fk(arms):
    report = []
    for arm in arms:
        fk = make_fk(arm)
        errors = []
        for q in (np.zeros(6), np.array([.2, -.4, .5, -.2, .3, -.1])):
            errors.append(float(np.max(abs(np.asarray(fk(q))-arm.chain.fk(q)@arm.offset()))))
        q = np.array([.2, -.4, .5, -.2, .3, -.1], dtype=np.float32)
        jac = np.asarray(jax.jacfwd(lambda x: fk(x)[:3, 3])(q))
        eps = 1e-4
        numerical = np.column_stack([((arm.chain.fk(q+np.eye(6)[j]*eps)@arm.offset())[:3, 3]
                                    -(arm.chain.fk(q-np.eye(6)[j]*eps)@arm.offset())[:3, 3])/(2*eps)
                                   for j in range(6)])
        error = float(np.max(abs(jac-numerical)))
        if max(errors)>2e-6 or error>2e-5:
            raise ValueError(('differentiable FK mismatch', errors, error))
        report.append(dict(max_matrix_error=max(errors), max_jacobian_error=error))
    return report

