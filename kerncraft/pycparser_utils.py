#!/usr/bin/env python3
"""Collection of functions that extend pycparser's functionality."""

import collections

from pycparser import c_ast


def clean_code(code, comments=True, macros=False, pragmas=False):
    """
    Naive comment and macro striping from source code

    :param comments: If True, all comments are stripped from code
    :param macros: If True, all macros are stripped from code
    :param pragmas: If True, all pragmas are stripped from code

    :return: cleaned code. Line numbers are preserved with blank lines,
    and multiline comments and macros are supported. BUT comment-like
    strings are (wrongfully) treated as comments.
    """
    if macros or pragmas:
        lines = code.split('\n')
        in_macro = False
        in_pragma = False
        for i in range(len(lines)):
            l = lines[i].strip()

            if macros and (l.startswith('#') and not l.startswith('#pragma') or in_macro):
                lines[i] = ''
                in_macro = l.endswith('\\')
            if pragmas and (l.startswith('#pragma') or in_pragma):
                lines[i] = ''
                in_pragma = l.endswith('\\')
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


def replace_id(ast, id_name, replacement):
    """
    Replace all matching ID nodes in ast (in-place), with replacement.

    :param id_name: name of ID node to match
    :param replacement: single or list of node to insert in replacement for ID node.
    """
    for a in ast:
        if isinstance(a, c_ast.ID) and a.name == id_name:
            # Check all attributes of ast
            for attr_name in dir(ast):
                # Exclude special and method attributes
                if attr_name.startswith('__') or callable(getattr(ast, attr_name)):
                    continue
                attr = getattr(ast, attr_name)
                # In case of direct match, just replace
                if attr is a:
                    setattr(ast, attr_name, replacement)
                # If contained in list replace occurrence with replacement
                if type(attr) is list:
                    for i, attr_element in enumerate(attr):
                        if attr_element is a:
                            if type(replacement) is list:
                                # If replacement is list, inject
                                attr[i:i+1] = replacement
                            else:
                                # otherwise replace
                                attr[i] = replacement
        else:
            replace_id(a, id_name, replacement)
