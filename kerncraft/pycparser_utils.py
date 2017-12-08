#!/usr/bin/env python3
"""Collection of functions that extend pycparser's functionality."""


def clean_code(code, comments=True, macros=False):
    """ Naive comment and macro striping from source code

        comments:
            If True, all comments are stripped from code

        macros:
            If True, all macros are stripped from code

        Returns cleaned code. Line numbers are preserved with blank lines,
        and multiline comments and macros are supported. BUT comments-like
        strings are (wrongfuly) treated as comments.
    """
    if macros:
        lines = code.split('\n')
        in_macro = False
        for i in range(len(lines)):
            l = lines[i].strip()

            if l.startswith('#') or in_macro:
                lines[i] = ''
                in_macro = l.endswith('\\')
        code = '\n'.join(lines)

    if comments:
        idx = 0
        comment_start = None
        while idx < len(code) - 1:
            if comment_start is None and code[idx:idx + 2] == '//':
                end_idx = code.find('\n', idx)
                code = code[:idx] + code[end_idx:]
                idx -= end_idx - idx
            elif comment_start is None and code[idx:idx + 2] == '/*':
                comment_start = idx
            elif comment_start is not None and code[idx:idx + 2] == '*/':
                code = (code[:comment_start] +
                        '\n' * code[comment_start:idx].count('\n') +
                        code[idx + 2:])
                idx -= idx - comment_start
                comment_start = None
            idx += 1

    return code
