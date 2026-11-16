# Binary Addition, Dyck-1, Unique Set, and Copy task generators.
import numpy as np
import torch

def _resolve_rng(rng=None):
    return np.random if rng is None else rng

def _rng_integers(rng, low, high=None, size=None):
    rng = _resolve_rng(rng)
    if hasattr(rng, 'integers'):
        return rng.integers(low, high, size=size)
    return rng.randint(low, high, size=size)

def _dyck_token_layout(num_bracket_types):
    separator_token = 2 * num_bracket_types
    pad_token = separator_token + 1
    eos_token = separator_token + 2
    ignore_token = separator_token + 3
    return (separator_token, pad_token, eos_token, ignore_token)

def _dyck_future_feasible(depth, steps_left, require_nonempty_suffix, current_max_depth, max_depth=None, exact_depth=None):
    if depth < 0:
        return False
    if max_depth is not None and current_max_depth > max_depth:
        return False
    if exact_depth is not None:
        if current_max_depth > exact_depth:
            return False
        if current_max_depth < exact_depth and depth + steps_left < exact_depth:
            return False
    if steps_left == 0:
        if require_nonempty_suffix and depth <= 0:
            return False
        return True
    return True

def _generate_dyck_prefix(length, num_bracket_types, *, max_depth=None, require_nonempty_suffix=True, exact_depth=None, max_attempts=256, rng=None):
    if length < 1:
        raise ValueError('Dyck prompt length must be at least 1.')
    if num_bracket_types < 1:
        raise ValueError('Dyck task requires at least one bracket type.')
    if max_depth is not None and max_depth < 1:
        raise ValueError('dyck_max_depth must be at least 1 when provided.')
    if exact_depth is not None and exact_depth < 1:
        raise ValueError('exact_depth must be at least 1 when provided.')
    if exact_depth is not None and exact_depth > length:
        raise ValueError('exact_depth cannot exceed the prompt length.')
    if max_depth is not None and exact_depth is not None and (exact_depth > max_depth):
        raise ValueError('exact_depth cannot exceed dyck_max_depth.')
    for _ in range(max_attempts):
        stack = []
        prefix_tokens = []
        current_max_depth = 0
        valid = True
        for pos in range(length):
            steps_left = length - pos - 1
            open_types = []
            can_close = False
            if max_depth is None or len(stack) < max_depth:
                next_depth = len(stack) + 1
                next_max_depth = max(current_max_depth, next_depth)
                if _dyck_future_feasible(next_depth, steps_left, require_nonempty_suffix, next_max_depth, max_depth=max_depth, exact_depth=exact_depth):
                    open_types = list(range(num_bracket_types))
            if stack:
                next_depth = len(stack) - 1
                if _dyck_future_feasible(next_depth, steps_left, require_nonempty_suffix, current_max_depth, max_depth=max_depth, exact_depth=exact_depth):
                    can_close = True
            move_types = []
            if open_types:
                move_types.append('open')
            if can_close:
                move_types.append('close')
            if not move_types:
                valid = False
                break
            move = move_types[int(_rng_integers(rng, 0, len(move_types)))]
            if move == 'open':
                bracket_type = int(_rng_integers(rng, 0, num_bracket_types))
                stack.append(bracket_type)
                current_max_depth = max(current_max_depth, len(stack))
                prefix_tokens.append(bracket_type)
            else:
                bracket_type = stack.pop()
                prefix_tokens.append(num_bracket_types + bracket_type)
        if not valid:
            continue
        if require_nonempty_suffix and len(stack) == 0:
            continue
        if exact_depth is not None and current_max_depth != exact_depth:
            continue
        return (prefix_tokens, list(stack), current_max_depth)
    raise RuntimeError(f'Failed to generate Dyck prefix after {max_attempts} attempts (length={length}, num_bracket_types={num_bracket_types}, max_depth={max_depth}, exact_depth={exact_depth}).')

def generate_prompt_matrix_copy(b, max_len, min_num_digits=1, max_num_digits=10, rng=None):
    batch_num_digits = _rng_integers(rng, min_num_digits, max_num_digits, size=(b, 1))
    num_digits = batch_num_digits.flatten()
    prompt_matrix = np.full((b, 2 * max_len), 3)
    y_matrix = np.full((b, 2 * max_len), 3)
    mask = np.full((b, 2 * max_len), 0)
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = _rng_integers(rng, low=0, high=2, size=num_digits[i])
        prompt_matrix[i, num_digits[i]] = 2
        y_matrix[i, :num_digits[i]] = 4
        y_matrix[i, num_digits[i]:2 * num_digits[i]] = prompt_matrix[i, :num_digits[i]]
        mask[i, num_digits[i]:] = 1
    return (torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask))

def binary_addition(a, b, return_intermediate=False):
    max_length = max(len(a), len(b))
    a = torch.tensor(np.flip(a).copy(), dtype=torch.int)
    b = torch.tensor(np.flip(b).copy(), dtype=torch.int)
    a = torch.cat((a, torch.zeros(max_length - len(a), dtype=torch.int)))
    b = torch.cat((b, torch.zeros(max_length - len(b), dtype=torch.int)))
    result = torch.zeros(max_length + 1, dtype=torch.int)
    carry = 0
    intermediate_results = []
    intermediate_carries = []
    for i in range(max_length):
        temp_sum = a[i] + b[i] + carry
        result[i] = temp_sum % 2
        carry = temp_sum // 2
        if return_intermediate:
            intermediate_results.append(result.clone())
            intermediate_carries.append(carry)
    result[max_length] = carry
    if return_intermediate:
        intermediate_results.append(result.clone())
        intermediate_carries.append(carry)
    final_result = reversed(result)
    if return_intermediate:
        return (final_result, intermediate_results, intermediate_carries)
    else:
        return final_result

def generate_prompt_matrix_addition(b, max_len, min_num_digits=1, max_num_digits=10, rng=None):
    batch_num_digits = _rng_integers(rng, min_num_digits, max_num_digits, size=(b, 1))
    num_digits = batch_num_digits.flatten()
    prompt_matrix = np.full((b, 3 * max_len), 3)
    y_matrix = np.full((b, 3 * max_len), 3)
    mask = np.full((b, 3 * max_len), 0)
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = _rng_integers(rng, low=0, high=2, size=num_digits[i])
        prompt_matrix[i, num_digits[i]] = 2
        prompt_matrix[i, num_digits[i] + 1:2 * num_digits[i] + 1] = _rng_integers(rng, low=0, high=2, size=num_digits[i])
        prompt_matrix[i, 2 * num_digits[i] + 1] = 5
        y_matrix[i, :2 * num_digits[i] + 1] = 4
        y_matrix[i, 2 * num_digits[i] + 1:3 * num_digits[i] + 2] = binary_addition(prompt_matrix[i, :num_digits[i]], prompt_matrix[i, num_digits[i] + 1:2 * num_digits[i] + 1])
        mask[i, 2 * num_digits[i] + 1:] = 1
    return (torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask))

def generate_prompt_matrix_dyck(b, max_len, min_num_digits=1, max_num_digits=10, num_bracket_types=1, max_depth=None, require_nonempty_suffix=True, exact_depth=None, rng=None):
    if exact_depth is not None and max_depth is None:
        max_depth = exact_depth
    batch_num_digits = _rng_integers(rng, min_num_digits, max_num_digits, size=(b, 1))
    num_digits = batch_num_digits.flatten()
    max_prompt_len = int(np.max(num_digits)) if len(num_digits) > 0 else max(1, max_num_digits - 1)
    (separator_token, pad_token, eos_token, ignore_token) = _dyck_token_layout(num_bracket_types)
    seq_len = 2 * max_prompt_len + 2
    prompt_matrix = np.full((b, seq_len), pad_token)
    y_matrix = np.full((b, seq_len), pad_token)
    mask = np.full((b, seq_len), 0)
    for i in range(b):
        prefix_len = int(num_digits[i])
        (prefix_tokens, stack, _) = _generate_dyck_prefix(prefix_len, num_bracket_types, max_depth=max_depth, require_nonempty_suffix=require_nonempty_suffix, exact_depth=exact_depth, rng=rng)
        suffix_tokens = [num_bracket_types + bracket_type for bracket_type in reversed(stack)]
        answer_tokens = suffix_tokens + [eos_token]
        prompt_matrix[i, :prefix_len] = np.array(prefix_tokens, dtype=np.int64)
        prompt_matrix[i, prefix_len] = separator_token
        y_matrix[i, :prefix_len + 1] = ignore_token
        answer_start = prefix_len + 1
        answer_end = answer_start + len(answer_tokens)
        y_matrix[i, answer_start:answer_end] = np.array(answer_tokens, dtype=np.int64)
        mask[i, answer_start:answer_end] = 1
    return (torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask))

def compute_ordered_array(arr):
    unique_nums = []
    for num in arr:
        if num not in unique_nums:
            unique_nums.append(num)
    return np.array(unique_nums)

def generate_prompt_matrix_dict(b, max_len, min_num_digits=1, max_num_digits=10, rng=None):
    batch_num_digits = _rng_integers(rng, min_num_digits, max_num_digits, size=(b, 1))
    num_digits = batch_num_digits.flatten()
    prompt_matrix = np.full((b, max_len * 2), 53)
    y_matrix = np.full((b, max_len * 2), 53)
    mask = np.full((b, max_len * 2), 0)
    for i in range(b):
        prompt_matrix[i, :num_digits[i]] = _rng_integers(rng, low=0, high=50, size=num_digits[i])
        prompt_matrix[i, num_digits[i]] = 51
        y_matrix[i, :num_digits[i]] = 52
        y_array = compute_ordered_array(prompt_matrix[i, :num_digits[i]])
        y_len = len(y_array)
        y_matrix[i, num_digits[i]:num_digits[i] + y_len] = np.copy(y_array)
        mask[i, num_digits[i]:] = 1
    return (torch.tensor(prompt_matrix), torch.tensor(batch_num_digits), torch.tensor(y_matrix), torch.tensor(mask))
