# Gated Retention ISS Stability

Created: 2026-05-04

Status: refinement of milestone-gated RetNet stability. This proof adds an
input-to-state stability view so the gate theorem does not merely say "do not
decay"; it also bounds state saturation under updates.

## 0. Model

Consider a projected protected state:

```text
s_t = g_t s_{t-1} + u_t
```

where:

```text
0 <= g_t <= 1
||u_t|| <= U_t
```

Let:

```text
epsilon_t = 1 - g_t
```

## 1. Exact Unrolling

For `T > i`:

```text
s_T =
(prod_{r=i+1}^T g_r) s_i
+ sum_{t=i+1}^T (prod_{r=t+1}^T g_r) u_t
```

The first term is retention of the milestone. The second term is accumulated
interference.

## 2. Preservation Term

If:

```text
sum_{r=i+1}^T epsilon_r <= eta
epsilon_r <= 1/2
```

then:

```text
prod_{r=i+1}^T g_r >= exp(-2 eta)
```

as in proof 20.

## 3. Interference Term

If the protected interval keeps `g_r >= 1 - epsilon`, then old additive updates
also remain. A crude bound is:

```text
||sum_{t=i+1}^T (prod_{r=t+1}^T g_r) u_t||
<= sum_{t=i+1}^T U_t
```

This explains state saturation. Preservation alone is not stability.

To get input-to-state stability, require either:

```text
1. finite protected window W
2. update projection ||P u_t|| small on protected subspace
3. overwrite/budget policy limiting active milestones
4. average contraction after relevance expires
```

## 4. ISS Bound With Expiring Protection

Assume after a protected window `W`, the gate returns to:

```text
g_t <= gamma_bar < 1
```

and `||u_t|| <= U`. Then after expiration:

```text
||s_T||
<= gamma_bar^{T-i-W} ||s_{i+W}||
 + U / (1 - gamma_bar)
```

This is the standard input-to-state stability form:

```text
state <= decayed initial condition + bounded input gain
```

## 5. Stability Condition For Milestone Policies

A milestone policy is stable only if it enforces:

```text
active protected windows <= B_time
projected update energy <= U_P
post-window contraction gamma_bar < 1
spam rate f_m low enough that windows do not overlap without bound
```

Otherwise:

```text
s_T = s_i + sum updates
```

and the system saturates even if gradients do not vanish.

## 6. Design Consequence

For a serious architecture, milestone gate should not be described as:

```text
make gamma close to 1 forever
```

It should be described as:

```text
temporarily preserve a protected subspace while enforcing ISS through bounded
windows, update projection, and contraction after relevance expires.
```

## 7. Experimental Obligations

Report:

```text
active milestone count
window length distribution
projected update norm in protected subspace
state norm growth over time
spam/false-positive milestone rate
module-drop effect
```
