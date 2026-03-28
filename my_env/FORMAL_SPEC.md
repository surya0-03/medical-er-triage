# Formal Safety Specification (Abstraction)

This document defines machine-checkable safety properties for the ER RL environment.

## State Variables

- `P`: patients dictionary keys
- `W`: waiting queue patient ids
- `B`: bed assignment patient ids
- `O`: delayed outcome patient ids
- `state(pid)`: lifecycle state for patient `pid`
- `icu_free`, `general_free`, `hallway_used`

## Safety Invariants

1. Partition invariant:
- `W`, `B`, and `O` are pairwise disjoint.

2. Membership/state invariant:
- `pid in W => state(pid) = waiting`
- `pid in B => state(pid) in {assigned_bed, in_treatment}`
- `pid in O => state(pid) = outcome_pending`

3. Resource invariant:
- `0 <= icu_free <= ICU_TOTAL`
- `0 <= general_free <= GENERAL_TOTAL`
- `0 <= hallway_used <= HALLWAY_CAP`
- `icu_free = ICU_TOTAL - count(icu beds in B)`
- `general_free = GENERAL_TOTAL - count(general beds in B)`
- `hallway_used = count(hallway beds in B)`

4. Capacity invariant:
- `|P| <= MAX_PATIENTS_TOTAL`
- `|O| <= MAX_DELAYED_OUTCOMES`

5. Progression invariant:
- In waiting state, a patient is either untriaged (`last_assigned_esi=None`) or triaged (`last_assigned_esi!=None`), and action mask partitions those sets.

## Transition Obligations

For each transition function `T in {reset, step phases, repair}`:

- Precondition: invariants hold at state `s`.
- Postcondition: invariants hold at state `T(s)`.

## Induction Skeleton

- Base case: prove invariants after `reset`.
- Inductive step: assume invariants before each `step`; prove each phase preserves invariants.
- Conclude invariants hold for all reachable steps.

## Model Checking Scope

For exhaustive checks, use bounded abstraction:

- Patients <= 4
- ICU beds <= 2, General beds <= 2, Hallway <= 1
- Horizon <= 12 steps

Check reachability of bad states:

- overlap(W,B) or overlap(W,O) or overlap(B,O)
- negative resources or cap overflow
- state/membership mismatch

## Runtime-Formal Bridge

The implementation's `_validate_state_consistency()` and `_assert_contracts()` are runtime realizations of these invariants. Hypothesis tests provide stochastic counterexample search against the same contracts.
