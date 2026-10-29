# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alpamayo history-preserving Edge driving prompt."""

DRIVING_CHAT_TEMPLATE = r"""{# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. #}
{# SPDX-License-Identifier: Apache-2.0 #}
{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '\n' -}}
{%- for content in message.content -%}
{%- if content.type == 'text' -%}{{- content.text -}}
{%- elif content.type == 'image' -%}{{- '<|vision_start|><|image_pad|><|vision_end|>' -}}
{%- elif content.type == 'trajectory' -%}{{- content.placeholder -}}
{%- endif -%}
{%- endfor -%}
{{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n<|cot_start|>' -}}{%- endif -%}
"""
