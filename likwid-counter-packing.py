#!/usr/bin/env python3
from pprint import pprint
import re
import string


def group_iterator(group):
    """
    Yild all groups of simple regex-like expression.

    The only special character is a dash (-), which take the preceding and the following chars to
    compute a range. If the range is non-sensical (e.g., b-a) it will be empty

    Example:
    >>> list(group_iterator('a-f'))
    ['a', 'b', 'c', 'd', 'e', 'f']
    >>> list(group_iterator('148'))
    ['1', '4', '8']
    >>> list(group_iterator('7-9ab'))
    ['7', '8', '9', 'a', 'b']
    >>> list(group_iterator('0B-A1'))
    ['0', '1']
    """
    ordered_chars = string.ascii_letters + string.digits
    tokenizer = ('(?P<seq>[a-zA-Z0-9]-[a-zA-Z0-9])|'
                 '(?P<chr>.)')
    for m in re.finditer(tokenizer, group):
        if m.group('seq'):
            start, sep, end = m.group('seq')
            for i in range(ordered_chars.index(start), ordered_chars.index(end)+1):
                yield ordered_chars[i]
        else:
            yield m.group('chr')


def register_options(regdescr):
    """
    Very reduced regular expressions for describing a group of registers.

    Only groups in square bracktes and unions with pipes (|) are supported.

    Examples:
    >>> list(register_options('PMC[0-3]'))
    ['PMC0', 'PMC1', 'PMC2', 'PMC3']
    >>> list(register_options('MBOX0C[01]'))
    ['MBOX0C0', 'MBOX0C1']
    >>> list(register_options('CBOX2C1'))
    ['CBOX2C1']
    >>> list(register_options('CBOX[0-3]C[01]'))
    ['CBOX0C0', 'CBOX0C1', 'CBOX1C0', 'CBOX1C1', 'CBOX2C0', 'CBOX2C1', 'CBOX3C0', 'CBOX3C1']
    >>> list(register_options('PMC[0-1]|PMC[23]'))
    ['PMC0', 'PMC1', 'PMC2', 'PMC3']

    """
    if not regdescr:
        yield None
    tokenizer = ('\[(?P<grp>[^]]+)\]|'
                 '(?P<chr>.)')
    for u in regdescr.split('|'):
        m = re.match(tokenizer, u)

        if m.group('grp'):
            current = group_iterator(m.group('grp'))
        else:
            current = [m.group('chr')]

        for c in current:
            if u[m.end():]:
                for r in register_options(u[m.end():]):
                    yield c + r
            else:
                yield c


def eventstr(event_tuple=None, event=None, register=None, parameters=None):
    """
    Return a LIKWID event string from an event tuple or keyword arguments.

    *event_tuple* may have two or three arguments: (event, register) or
    (event, register, parameters)

    Keyword arguments will be overwritten by *event_tuple*.

    >>> eventstr(('L1D_REPLACEMENT', 'PMC0', None))
    'L1D_REPLACEMENT:PMC0'
    >>> eventstr(('L1D_REPLACEMENT', 'PMC0'))
    'L1D_REPLACEMENT:PMC0'
    >>> eventstr(('MEM_UOPS_RETIRED_LOADS', 'PMC3', {'EDGEDETECT': None, 'THRESHOLD': 2342}))
    'MEM_UOPS_RETIRED_LOADS:PMC3:EDGEDETECT:THRESHOLD=0x926'
    >>> eventstr(event='DTLB_LOAD_MISSES_WALK_DURATION', register='PMC3')
    'DTLB_LOAD_MISSES_WALK_DURATION:PMC3'
    """
    if len(event_tuple) == 3:
        event, register, parameters = event_tuple
    elif len(event_tuple) == 2:
        event, register = event_tuple
    event_dscr = [event, register]

    if parameters:
        for k, v in sorted(event_tuple[2].items()):  # sorted for reproducability
            if type(v) is int:
                k += "={}".format(hex(v))
            event_dscr.append(k)
    return ":".join(event_dscr)


def build_minimal_runs(events):
    """Compile list of minimal runs for given events."""
    # Eliminate multiples
    events = [e for i, e in enumerate(events) if events.index(e) == i]

    # Build list of runs per register group
    scheduled_runs = {}
    scheduled_events = []
    cur_run = 0
    while len(scheduled_events) != len(events):
        for event_tpl in events:
            event, registers, parameters = event_tpl
            # Skip allready scheduled events
            if event_tpl in scheduled_events:
                continue
            # Compile explicit list of possible register locations
            for possible_reg in register_options(registers):
                # Schedule in current run, if register is not yet in use
                s = scheduled_runs.setdefault(cur_run, {})
                if possible_reg not in s:
                    s[possible_reg] = (event, possible_reg, parameters)
                    # ban from further scheduling attempts
                    scheduled_events.append(event_tpl)
                    break
        cur_run += 1

    # Collaps all register dicts to single runs
    runs = [list(v.values()) for v in scheduled_runs.values()]

    return runs


if __name__ == '__main__':
    requested = [
        ('L1D_REPLACEMENT', 'PMC[0-3]', None),
        ('BR_INST_RETIRED_ALL_BRANCHES', 'PMC[0-3]', None),
        ('BR_MISP_RETIRED_ALL_BRANCHES', 'PMC[0-3]', None),
        ('MEM_UOPS_RETIRED_LOADS', 'PMC[0-3]', None),
        ('MEM_UOPS_RETIRED_STORES', 'PMC[0-3]', None),
        ('DTLB_LOAD_MISSES_CAUSES_A_WALK', 'PMC[0-3]', None),
        ('DTLB_STORE_MISSES_CAUSES_A_WALK', 'PMC[0-3]', None),
        ('DTLB_LOAD_MISSES_WALK_DURATION', 'PMC[0-3]', None),
        ('DTLB_STORE_MISSES_WALK_DURATION', 'PMC[0-3]', None),
        ('CYCLE_ACTIVITY_CYCLES_L2_PENDING', 'PMC2', None),
        ('MEM_UOPS_RETIRED_LOADS', 'PMC[0-3]', {'EDGEDETECT': None}),
        ('MEM_UOPS_RETIRED_LOADS', 'PMC[0-3]', {'EDGEDETECT': None, 'THRESHOLD': 2342}),
        ('MEM_UOPS_RETIRED_LOADS', 'PMC[0-3]', {'EDGEDETECT': None, 'THRESHOLD': 0x926}),
        ('CPU_CLOCK_UNHALTED_THREAD_P', 'PMC[0-3]', {'THRESHOLD': 0x2, 'INVERT': None}),
        ('CAS_COUNT_RD', 'MBOX0C[01]', None),
        ('CAS_COUNT_WR', 'MBOX0C[01]', None),
        ('CAS_COUNT_RD', 'MBOX1C[01]', None),
        ('CAS_COUNT_WR', 'MBOX1C[01]', None),
        ('CAS_COUNT_RD', 'MBOX2C[01]', None),
        ('CAS_COUNT_WR', 'MBOX2C[01]', None),
        ('CAS_COUNT_RD', 'MBOX3C[01]', None),
        ('CAS_COUNT_WR', 'MBOX3C[01]', None),
        ('CAS_COUNT_RD', 'MBOX4C[01]', None),
        ('CAS_COUNT_WR', 'MBOX4C[01]', None),
        ('CAS_COUNT_RD', 'MBOX5C[01]', None),
        ('CAS_COUNT_WR', 'MBOX5C[01]', None),
        ('CAS_COUNT_RD', 'MBOX6C[01]', None),
        ('CAS_COUNT_WR', 'MBOX6C[01]', None),
        ('CAS_COUNT_RD', 'MBOX7C[01]', None),
        ('CAS_COUNT_WR', 'MBOX7C[01]', None),
        ('LLC_LOOKUP_ANY', 'CBOX0C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX1C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX2C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX3C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX4C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX5C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX6C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX7C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX8C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX9C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX10C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX11C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX12C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX13C1', {'STATE': 0x1}),
        ('LLC_LOOKUP_ANY', 'CBOX14C1', {'STATE': 0x1}),
    ]

    print(len(requested))
    runs = build_minimal_runs(requested)
    pprint([[eventstr(e) for e in r] for r in runs])
    print(len(runs))
